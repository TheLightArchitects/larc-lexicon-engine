//! Extraction of genuinely human-authored text from agent-session transcripts.
//!
//! # Why this module exists
//!
//! A Claude Code session transcript (`~/.claude/projects/<slug>/<uuid>.jsonl`)
//! stores several materially different things under the same `"type": "user"`
//! tag, and only some of it is text a human actually wrote. Measured over
//! 14,425 `type == "user"` entries across one real nine-session project:
//!
//! | `origin.kind` | `content` | count | what it really is |
//! |---|---|---:|---|
//! | absent | array | 11,537 | `tool_result` blocks re-injected as user turns |
//! | absent | string | 2,226 | harness/hook-injected synthetic turns |
//! | `task-notification` | string | 342 | background-task completion notices |
//! | `human` | string | 304 | **the author's own typed turns** |
//! | `human` | array | 10 | **the author's text, plus an attached image** |
//! | `peer` | string | 6 | messages relayed *from another agent session* |
//!
//! The naive filter (`type == "user"` and a string body) captures 2,878 of
//! those entries, of which **2,574 — 89% — are not the author's writing**. A
//! lexicon built that way attributes harness boilerplate and other agents'
//! prose to the human: the exact failure mode [`crate::schema::Confidence`]
//! exists to prevent, arriving through the back door of the ingest path.
//!
//! So authorship is decided by `origin.kind == "human"` alone, and nothing
//! looser — a mislabeled sample is worse than a missing one.
//!
//! Body shape is then a *recall* question, not an authorship one. Once an
//! entry is known to be human-authored, an array body is not a tool result
//! (those never carry a `human` origin) but a message with an attachment, and
//! its `text` blocks are ordinary authored prose — often the most opinionated
//! kind, since they are reactions to a screenshot. Dropping them for their
//! shape would discard ~3% of the corpus for no reason, so they are joined and
//! kept.
//!
//! # Harness boilerplate
//!
//! Even a correctly-identified human turn is not clean text. The harness
//! appends injected XML blocks (`<system-reminder>`, slash-command echoes,
//! captured local-command output) to the literal characters the human typed.
//! Feeding that into [`crate::compute_linguistic_profile`] corrupts every
//! measurement it makes — dash rate, sentence length, type-token ratio — with
//! boilerplate nobody authored. [`strip_harness_blocks`] removes it first.
//!
//! # Pasted material
//!
//! A fourth contamination vector, distinct from the three above: a turn can
//! be genuinely human-authored, harness-clean, and *still* not be prose the
//! author composed — because a chat message can contain a pasted crash
//! report, job posting, or API/tool schema dump. Measured on one real
//! 248-sample ingest, 3 turns (1.2%) carried 51.2% of the corpus's words —
//! a macOS crash report, a job description, and an MCP tool schema — enough
//! to dominate every word-weighted metric computed over the batch.
//!
//! [`flag_pasted_content`] flags candidates with two independent, general
//! structural signals rather than matching the wording of those three
//! specific documents, which would never generalize to a different corpus:
//! a robust length-outlier test against the batch's own word-count
//! distribution, and detection of label:value / separator-rule line shapes
//! characteristic of rendered documents rather than typed prose. Matching
//! [`crate::schema::Confidence`]'s rule that suspect data is *labeled*, not
//! silently dropped, this only flags — nothing here removes a sample; a
//! consumer decides what to do with the signal.

use serde::Deserialize;

use crate::metrics::word_count;

/// One human-authored turn recovered from a transcript, already stripped of
/// harness boilerplate and guaranteed non-empty after trimming.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedTurn {
    /// The human's text, with harness-injected blocks removed.
    pub text: String,
    /// Originating session id, when the entry carried one — becomes
    /// `SourceRef::session_id` so a sample traces back to its exact session.
    pub session_id: Option<String>,
    /// Working directory recorded on the entry — the best available proxy for
    /// which project the turn was written in.
    pub cwd: Option<String>,
    /// RFC 3339 timestamp recorded on the entry, if present.
    pub timestamp: Option<String>,
}

/// XML-ish blocks the harness injects into, or around, human turns. Each is
/// removed wholesale (open tag through close tag) before profiling.
///
/// `cross-session-message` is included defensively: such entries are normally
/// `origin.kind == "peer"` and already excluded by the author filter, but a
/// quoted one appearing inside a human turn is still not the human's prose.
const HARNESS_BLOCK_TAGS: &[&str] = &[
    "system-reminder",
    "cross-session-message",
    "local-command-caveat",
    "local-command-stdout",
    "command-name",
    "command-message",
    "command-args",
];

/// Minimal projection of a transcript line — every field optional because
/// transcript schemas drift across harness versions, and a missing field must
/// degrade to "skip this entry", never to a parse failure that aborts the file.
#[derive(Deserialize)]
struct RawEntry {
    #[serde(rename = "type")]
    entry_type: Option<String>,
    origin: Option<RawOrigin>,
    message: Option<RawMessage>,
    #[serde(rename = "sessionId")]
    session_id: Option<String>,
    cwd: Option<String>,
    timestamp: Option<String>,
}

#[derive(Deserialize)]
struct RawOrigin {
    kind: Option<String>,
}

#[derive(Deserialize)]
struct RawMessage {
    /// `serde_json::Value` rather than `String`: a turn's body is a bare string
    /// for plain text and an array of typed blocks when anything is attached.
    content: Option<serde_json::Value>,
}

/// Flatten a message body to its authored text.
///
/// A string body is the text itself. An array body is a block list — keep the
/// `text` blocks (joined in order) and ignore `image` and other non-text
/// blocks, which carry no prose to measure.
fn body_text(content: &serde_json::Value) -> Option<String> {
    match content {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Array(blocks) => {
            let joined = blocks
                .iter()
                .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("text"))
                .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("\n");
            (!joined.trim().is_empty()).then_some(joined)
        }
        _ => None,
    }
}

/// Remove `[Image #N]` attachment markers, which the harness writes into the
/// text body itself. They are references to an attachment, not authored words,
/// and their brackets and digits would otherwise perturb the metrics.
fn strip_attachment_markers(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("[Image #") {
        let after = &rest[start..];
        match after.find(']') {
            // `.count() > 8`, not just the `.all()` below: on an empty
            // iterator (a bare "[Image #]" with zero digits) `.all()` is
            // vacuously true, which would otherwise strip literal text
            // that merely mentions the marker format with no number in it.
            Some(end)
                if after[..end].chars().count() > 8
                    && after[..end].chars().skip(8).all(|c| c.is_ascii_digit()) =>
            {
                out.push_str(&rest[..start]);
                rest = &after[end + 1..];
            }
            _ => {
                let consumed = start + "[Image #".len();
                out.push_str(&rest[..consumed]);
                rest = &rest[consumed..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// If `after_name` (the text immediately following `<tag`) opens with a
/// self-closing form — `/>` or `attr="x"/>` — return the byte offset just
/// past its `>`. `None` means this is a normal open tag (or malformed).
fn self_closing_end(after_name: &str) -> Option<usize> {
    let gt = after_name.find('>')?;
    after_name[..gt].ends_with('/').then_some(gt + 1)
}

/// Remove one `<tag ...>...</tag>` block, including its delimiters, or one
/// self-closing `<tag ... />`, everywhere either appears.
///
/// Matches an open tag only when the character after the tag name is `>`,
/// whitespace, or `/`, so stripping `command` never eats a `<commanding>`
/// element. A self-closing tag removes only itself — searching for a
/// `</tag>` close it will never have would otherwise consume everything
/// after it. A genuinely unclosed *paired* open tag still drops the
/// remainder of the string: harness blocks are appended at the end, so a
/// truncated one has no authored text after it.
fn strip_tag_block(text: &str, tag: &str) -> String {
    let open_prefix = format!("<{tag}");
    let close = format!("</{tag}>");
    let mut out = String::with_capacity(text.len());
    let mut rest = text;

    loop {
        let Some(start) = rest.find(&open_prefix) else {
            out.push_str(rest);
            return out;
        };

        let after_name = &rest[start + open_prefix.len()..];

        if let Some(end) = self_closing_end(after_name) {
            out.push_str(&rest[..start]);
            rest = &after_name[end..];
            continue;
        }

        let is_tag =
            after_name.starts_with('>') || after_name.starts_with(|c: char| c.is_whitespace());
        if !is_tag {
            // A longer element name that merely shares this prefix — keep it.
            let consumed = start + open_prefix.len();
            out.push_str(&rest[..consumed]);
            rest = &rest[consumed..];
            continue;
        }

        out.push_str(&rest[..start]);
        match find_balanced_close(after_name, &open_prefix, &close) {
            Some(rel_end) => rest = &after_name[rel_end + close.len()..],
            None => return out,
        }
    }
}

/// Find the close tag matching the open tag whose contents start at
/// `after_name`, treating a nested instance of the *same* open tag as
/// increasing nesting depth rather than ending the block at the first
/// `</tag>` found. Without this, a block whose content happens to contain
/// another instance of the same tag — a pasted transcript excerpt quoting
/// one, for example — closes at the inner tag, leaving a dangling
/// `</tag>` string and the unremoved tail of the outer block in the
/// output. A nested *self-closing* instance of the same tag doesn't open a
/// paired block, so it's skipped rather than counted.
fn find_balanced_close(after_name: &str, open_prefix: &str, close: &str) -> Option<usize> {
    let mut depth = 1u32;
    let mut pos = 0usize;
    loop {
        let next_open = after_name[pos..].find(open_prefix).map(|i| pos + i);
        let next_close = after_name[pos..].find(close).map(|i| pos + i);
        match (next_open, next_close) {
            (Some(o), Some(c)) if o < c => {
                let nested_after_name = &after_name[o + open_prefix.len()..];
                pos = o + open_prefix.len();
                if self_closing_end(nested_after_name).is_none() {
                    depth += 1;
                }
            }
            (_, Some(c)) => {
                depth -= 1;
                if depth == 0 {
                    return Some(c);
                }
                pos = c + close.len();
            }
            (_, None) => return None,
        }
    }
}

/// Strip every harness-injected block from a raw turn and trim the result.
///
/// Intended for any text captured out of an agent harness before it is
/// profiled or stored, not just transcripts.
pub fn strip_harness_blocks(text: &str) -> String {
    let mut out = text.to_string();
    for tag in HARNESS_BLOCK_TAGS {
        out = strip_tag_block(&out, tag);
    }
    out.trim().to_string()
}

/// Extract every human-authored turn from the contents of one Claude Code
/// session `.jsonl` file.
///
/// Malformed lines are skipped rather than propagated: transcripts are appended
/// to live and the final line of an in-progress session is routinely a partial
/// write. One truncated line must not cost the other several hundred turns.
pub fn extract_human_turns(jsonl: &str) -> Vec<ExtractedTurn> {
    jsonl
        .lines()
        .filter_map(|line| serde_json::from_str::<RawEntry>(line).ok())
        .filter_map(human_turn_from_entry)
        .collect()
}

fn human_turn_from_entry(entry: RawEntry) -> Option<ExtractedTurn> {
    if entry.entry_type.as_deref() != Some("user") {
        return None;
    }
    // Strict: only an explicit `human` origin qualifies. `peer` is another
    // agent, `task-notification` is the harness, and a *missing* origin is a
    // synthesized turn — none is the author's writing, and none may be guessed
    // into inclusion.
    if entry.origin.and_then(|o| o.kind).as_deref() != Some("human") {
        return None;
    }
    let text = body_text(&entry.message.and_then(|m| m.content)?)?;

    let text = strip_harness_blocks(&strip_attachment_markers(&text));
    if text.is_empty() {
        return None;
    }

    Some(ExtractedTurn {
        text,
        session_id: entry.session_id,
        cwd: entry.cwd,
        timestamp: entry.timestamp,
    })
}

/// A structural signal that one turn's text may be pasted material rather
/// than prose composed as a chat message. Purely advisory: nothing in this
/// crate excludes a sample on this basis by itself. Call [`PasteSignal::any`]
/// to get a single flag, or inspect the two fields to see which signal fired.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PasteSignal {
    /// Word count is a robust statistical outlier against the rest of the
    /// batch it was flagged alongside (see [`flag_pasted_content`]).
    pub length_outlier: bool,
    /// The text contains label:value lines or long separator-rule runs
    /// characteristic of a rendered document (crash report, log, API/tool
    /// schema) rather than typed prose.
    pub structural_markup: bool,
}

impl PasteSignal {
    pub fn any(&self) -> bool {
        self.length_outlier || self.structural_markup
    }
}

/// Below this batch size there isn't enough data for quartiles to mean
/// anything, so length-outlier detection falls back to a fixed backstop
/// instead (Algorithm Baseline T0.3 — default to the most robust general
/// approach when no data-driven method applies, and document the gap).
const MIN_BATCH_FOR_QUARTILES: usize = 8;

/// Word count beyond which a turn is flagged regardless of batch shape, used
/// only when the batch is too small for quartile-based detection. An order
/// of magnitude past the longest median turn length measured across two very
/// different registers in this crate's own corpora (12 and 21 words) —
/// deliberately generous, since a false positive here only adds a warning,
/// while a false negative lets a genuine outlier through uninspected.
const SMALL_BATCH_LENGTH_BACKSTOP: usize = 300;

/// Robust length-outlier test using Tukey's far-outlier fence
/// (`> Q3 + 3 * IQR`) over the batch's own word-count distribution, rather
/// than a fixed word-count guess that wouldn't generalize across users or
/// registers. The 3x (not the conventional 1.5x "outlier") multiplier is
/// deliberately conservative — this only flags for review, so it should
/// catch documents that dominate the batch, not merely someone's longer
/// message of the day.
fn length_outliers(word_counts: &[usize]) -> Vec<bool> {
    if word_counts.len() < MIN_BATCH_FOR_QUARTILES {
        return word_counts
            .iter()
            .map(|&n| n > SMALL_BATCH_LENGTH_BACKSTOP)
            .collect();
    }

    let mut sorted = word_counts.to_vec();
    sorted.sort_unstable();
    let quantile = |q: f64| -> f64 {
        let idx = q * (sorted.len() - 1) as f64;
        let lo = idx.floor() as usize;
        let hi = idx.ceil() as usize;
        if lo == hi {
            sorted[lo] as f64
        } else {
            let frac = idx - lo as f64;
            sorted[lo] as f64 * (1.0 - frac) + sorted[hi] as f64 * frac
        }
    };
    let q1 = quantile(0.25);
    let q3 = quantile(0.75);
    let iqr = q3 - q1;
    let fence = q3 + 3.0 * iqr;

    word_counts.iter().map(|&n| n as f64 > fence).collect()
}

/// A line consisting mostly of one repeated separator character, the shape
/// crash reports and log dumps use for section rules (`------- ... -------`).
fn is_separator_rule(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.chars().count() < 10 {
        return false;
    }
    for c in ['-', '=', '─', '*', '_'] {
        if trimmed.chars().filter(|&ch| ch == c).count() as f32 / trimmed.chars().count() as f32
            > 0.8
        {
            return true;
        }
    }
    false
}

/// A `Label: value` line — the shape of crash-report fields (`Code Type:`,
/// `Exception Type:`), tool/API schema dumps (`Tool name:`, `Full name:`),
/// and structured logs generally. General by construction: it matches the
/// *shape* of a rendered document's metadata field, not any specific
/// document's wording.
///
/// The label is capped at 3 words / 24 characters — real metadata field
/// names are that short by convention. Without this cap, an ordinary
/// sentence that merely contains a colon partway through ("Corso wanted to
/// run this by you Canonical copy: …") has an all-alphabetic, capitalized
/// prefix and would otherwise match; a genuine label never runs that long.
fn is_label_value_line(line: &str) -> bool {
    let trimmed = line.trim_start();
    let Some(colon) = trimmed.find(':') else {
        return false;
    };
    let label = &trimmed[..colon];
    let after = trimmed[colon + 1..].trim_start();
    (1..=24).contains(&label.chars().count())
        && label.split_whitespace().count() <= 3
        && label.chars().next().is_some_and(|c| c.is_uppercase())
        && label
            .chars()
            .all(|c| c.is_alphanumeric() || " /_'-".contains(c))
        && !after.is_empty()
}

/// True when at least 3 lines are separator rules or label:value lines, or
/// when such lines make up more than a quarter of all non-blank lines —
/// either is enough structural document-shape to warrant a flag on text of
/// any length, not just very long pastes.
pub fn detect_structural_markup(text: &str) -> bool {
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    if lines.is_empty() {
        return false;
    }
    let hits = lines
        .iter()
        .filter(|l| is_separator_rule(l) || is_label_value_line(l))
        .count();
    hits >= 3 || (hits as f32 / lines.len() as f32) > 0.25
}

/// Flag which texts in a batch look like pasted material rather than typed
/// prose. Returns one [`PasteSignal`] per input, in order. See the module
/// docs for why this exists and why it only flags rather than excludes.
pub fn flag_pasted_content(texts: &[&str]) -> Vec<PasteSignal> {
    let word_counts: Vec<usize> = texts.iter().map(|t| word_count(t) as usize).collect();
    let outliers = length_outliers(&word_counts);
    texts
        .iter()
        .zip(outliers)
        .map(|(text, length_outlier)| PasteSignal {
            length_outlier,
            structural_markup: detect_structural_markup(text),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // These are synthetic and shaped like — but not copies of — the three
    // real documents that motivated this module, so passing here is
    // evidence the detector generalizes rather than having been fit to
    // known text.

    #[test]
    fn crash_report_shaped_text_is_flagged_by_structural_markup() {
        let text = "-------------------------------------\n\
                     Translated Report\n\
                     -------------------------------------\n\
                     Process: SomeApp [1234]\n\
                     Path: /Applications/SomeApp.app/Contents/MacOS/SomeApp\n\
                     Identifier: com.example.someapp\n\
                     Version: 4.2.0\n\
                     Code Type: ARM-64\n\
                     Exception Type: EXC_BAD_ACCESS\n";
        assert!(detect_structural_markup(text));
    }

    #[test]
    fn tool_schema_shaped_text_is_flagged_by_structural_markup() {
        let text = "Tool name: search\n\
                     Full name: mcp__example__search\n\
                     Description: Searches the index\n\
                     Parameter: query (string, required)\n\
                     Parameter: limit (integer, optional)\n";
        assert!(detect_structural_markup(text));
    }

    #[test]
    fn ordinary_directive_prose_is_not_flagged_by_structural_markup() {
        let text = "Can you check the deploy logs before pushing to main? \
                     I think the readability formula might be off by a bit, \
                     but the fix should be simple once we find it.";
        assert!(!detect_structural_markup(text));
    }

    #[test]
    fn a_genuinely_long_message_alone_in_a_small_batch_is_not_flagged() {
        // Small-batch backstop only fires for extreme length (300+ words),
        // not merely "longer than usual" — a real detailed message must
        // survive alongside a couple of short ones.
        let long_but_real = "So the plan is: first refactor the parser to \
            separate tokenization from validation, then add a proper error \
            type instead of returning strings, then wire up the new tests \
            we discussed, and finally update the README to reflect the new \
            module layout. I want to do this in that order specifically so \
            each step is independently reviewable.";
        let batch = ["push it", "run round 2", long_but_real];
        let flags = flag_pasted_content(&batch);
        assert!(!flags[2].length_outlier, "flagged a genuine long message");
    }

    #[test]
    fn an_extreme_length_outlier_is_flagged_in_a_small_batch() {
        let huge = "word ".repeat(500);
        let batch = ["push it", "run round 2", huge.as_str()];
        let flags = flag_pasted_content(&batch);
        assert!(flags[2].length_outlier);
        assert!(!flags[0].length_outlier);
        assert!(!flags[1].length_outlier);
    }

    #[test]
    fn quartile_based_outlier_detection_fires_on_a_larger_batch() {
        // 10 ordinary short turns plus one that is ~30x the rest.
        let mut batch: Vec<String> = (0..10).map(|i| format!("turn number {i}")).collect();
        batch.push("word ".repeat(300));
        let refs: Vec<&str> = batch.iter().map(|s| s.as_str()).collect();
        let flags = flag_pasted_content(&refs);
        assert!(flags[10].length_outlier);
        assert!(flags[..10].iter().all(|f| !f.length_outlier));
    }

    #[test]
    fn paste_signal_any_is_true_when_either_signal_fires() {
        let neither = PasteSignal {
            length_outlier: false,
            structural_markup: false,
        };
        let one = PasteSignal {
            length_outlier: true,
            structural_markup: false,
        };
        assert!(!neither.any());
        assert!(one.any());
    }

    #[test]
    fn peer_and_tool_result_turns_are_never_attributed_to_the_human() {
        let jsonl = r#"
{"type":"user","origin":{"kind":"human"},"message":{"role":"user","content":"Ship the CLI."},"sessionId":"s1","cwd":"/repo","timestamp":"2026-09-06T14:00:00Z"}
{"type":"user","origin":{"kind":"peer"},"message":{"role":"user","content":"Relayed from another agent."}}
{"type":"user","message":{"role":"user","content":[{"type":"tool_result","content":"ok"}]}}
{"type":"user","message":{"role":"user","content":"Harness-synthesized turn with no origin."}}
{"type":"assistant","origin":{"kind":"human"},"message":{"role":"assistant","content":"Not a user turn."}}
"#;
        let turns = extract_human_turns(jsonl);
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].text, "Ship the CLI.");
        assert_eq!(turns[0].session_id.as_deref(), Some("s1"));
    }

    #[test]
    fn human_turns_with_an_attachment_keep_their_text_blocks() {
        // A screenshot reaction: `origin.kind` is human, but the body is a
        // block array. The prose in it is still the author's.
        let jsonl = r#"{"type":"user","origin":{"kind":"human"},"message":{"content":[{"type":"text","text":"I am not happy with the TUI. [Image #4]"},{"type":"image","source":{"data":"..."}}]}}"#;
        let turns = extract_human_turns(jsonl);
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].text, "I am not happy with the TUI.");
    }

    #[test]
    fn task_notification_origin_is_not_the_author() {
        let jsonl = r#"{"type":"user","origin":{"kind":"task-notification"},"message":{"content":"Background task finished."}}"#;
        assert!(extract_human_turns(jsonl).is_empty());
    }

    #[test]
    fn attachment_markers_are_removed_but_similar_text_is_kept() {
        assert_eq!(
            strip_attachment_markers("see [Image #12] here"),
            "see  here"
        );
        assert_eq!(
            strip_attachment_markers("an [Image #x] literal"),
            "an [Image #x] literal"
        );
    }

    /// A bare `[Image #]` with zero digits must be kept: `.all()` over the
    /// (empty) digit-check iterator is vacuously true, so without an
    /// explicit non-empty check this would be wrongly stripped as if it
    /// were a real marker, deleting literal text that merely mentions the
    /// marker format.
    #[test]
    fn bare_marker_with_no_digits_is_not_stripped() {
        assert_eq!(
            strip_attachment_markers("the format is [Image #] then digits"),
            "the format is [Image #] then digits"
        );
    }

    #[test]
    fn harness_blocks_are_stripped_from_human_text() {
        let raw = "Real sentence here.\n<system-reminder>\nInjected noise — with an em dash.\n</system-reminder>";
        assert_eq!(strip_harness_blocks(raw), "Real sentence here.");
    }

    #[test]
    fn tag_stripping_does_not_eat_longer_element_names() {
        let raw = "<commanding>keep me</commanding>";
        assert_eq!(
            strip_harness_blocks(raw),
            "<commanding>keep me</commanding>"
        );
    }

    #[test]
    fn unclosed_harness_block_drops_only_the_trailing_remainder() {
        let raw = "Authored text.<system-reminder>truncated injection";
        assert_eq!(strip_harness_blocks(raw), "Authored text.");
    }

    #[test]
    fn self_closing_harness_tag_removes_only_itself() {
        let raw = "Real words. <command-args/> more real words that should be kept";
        assert_eq!(
            strip_harness_blocks(raw),
            "Real words.  more real words that should be kept"
        );
    }

    /// A harness block whose own content quotes another instance of the
    /// same tag — plausible for this crate specifically, since a user
    /// working on transcript-processing tooling might paste an excerpt
    /// that itself contains a `<system-reminder>` — must close at its own
    /// matching close tag, not at the inner one. Closing early would leave
    /// a dangling `</system-reminder>` string and the outer block's tail
    /// in the output.
    #[test]
    fn harness_block_with_a_nested_same_tag_instance_closes_at_the_outer_tag() {
        let raw = "<system-reminder>outer <system-reminder>inner</system-reminder> tail</system-reminder> more real prose";
        assert_eq!(strip_harness_blocks(raw), "more real prose");
    }

    /// A nested *self-closing* instance of the same tag name doesn't open a
    /// paired block, so it must not consume a depth level — otherwise the
    /// real close tag would look one level too deep and the block would
    /// swallow real text past its actual end.
    #[test]
    fn nested_self_closing_instance_of_the_same_tag_does_not_add_depth() {
        let raw =
            "<system-reminder>outer <system-reminder/> tail</system-reminder> more real prose";
        assert_eq!(strip_harness_blocks(raw), "more real prose");
    }

    #[test]
    fn malformed_lines_are_skipped_without_losing_the_rest() {
        let jsonl = concat!(
            "{\"type\":\"user\",\"origin\":{\"kind\":\"human\"},\"message\":{\"content\":\"First.\"}}\n",
            "{not valid json\n",
            "{\"type\":\"user\",\"origin\":{\"kind\":\"human\"},\"message\":{\"content\":\"Second.\"}}\n"
        );
        let turns = extract_human_turns(jsonl);
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[1].text, "Second.");
    }

    #[test]
    fn turns_that_are_only_harness_boilerplate_are_dropped() {
        let jsonl = "{\"type\":\"user\",\"origin\":{\"kind\":\"human\"},\"message\":{\"content\":\"<system-reminder>all noise</system-reminder>\"}}";
        assert!(extract_human_turns(jsonl).is_empty());
    }
}
