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

use serde::Deserialize;

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
            Some(end) if after[..end].chars().skip(8).all(|c| c.is_ascii_digit()) => {
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

/// Remove one `<tag ...>...</tag>` block, including its delimiters, everywhere
/// it appears.
///
/// Matches an open tag only when the character after the tag name is `>` or
/// whitespace, so stripping `command` never eats a `<commanding>` element. An
/// unclosed open tag drops the remainder of the string: harness blocks are
/// appended at the end, so a truncated one has no authored text after it.
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
        let is_tag = after_name.starts_with('>')
            || after_name.starts_with(|c: char| c.is_whitespace())
            || after_name.starts_with('/');
        if !is_tag {
            // A longer element name that merely shares this prefix — keep it.
            let consumed = start + open_prefix.len();
            out.push_str(&rest[..consumed]);
            rest = &rest[consumed..];
            continue;
        }

        out.push_str(&rest[..start]);
        match after_name.find(&close) {
            Some(rel_end) => rest = &after_name[rel_end + close.len()..],
            None => return out,
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

#[cfg(test)]
mod tests {
    use super::*;

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
