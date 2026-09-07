//! Rendering the `larc style-guide` markdown brief — an author's distilled
//! `VoicePattern`s grouped Replicate/Avoid, each with real verbatim quotes,
//! shaped to be pasted directly into an LLM system prompt.
//!
//! Needs no storage or embeddings — it's a pure function of already-loaded
//! `VoicePattern`/`VoiceSample` data — so it lives in core rather than
//! behind `sqlite-backend`, and any caller with its own `LexiconEngine`
//! backend (or an in-process embedder, like a coding agent's tool registry)
//! can call it directly instead of reimplementing the same markdown shape.

use std::collections::HashMap;

use uuid::Uuid;

use crate::{PatternCategory, VoicePattern, VoiceSample};

const MAX_QUOTES_PER_PATTERN: usize = 3;
const MAX_QUOTE_CHARS: usize = 180;

/// Truncate `s` to `max` chars, appending an ellipsis if anything was cut.
fn truncate_with_ellipsis(s: &str, max: usize) -> String {
    let truncated: String = s.chars().take(max).collect();
    if s.chars().count() > max {
        format!("{truncated}…")
    } else {
        truncated
    }
}

/// Human-readable heading for a `PatternCategory` — the enum's `Debug`
/// output (`SentenceConstruction`, `AntiPattern`) reads fine in `larc
/// patterns list` but not as a markdown section header.
fn category_heading(c: PatternCategory) -> &'static str {
    match c {
        PatternCategory::Opener => "Opener",
        PatternCategory::SentenceConstruction => "Sentence Construction",
        PatternCategory::ToneRule => "Tone",
        PatternCategory::Vocabulary => "Vocabulary",
        PatternCategory::AntiPattern => "Anti-Pattern",
    }
}

/// Collapse a sample's text to one line and cap its length, so a long
/// pasted turn doesn't dominate the brief the way it dominated the
/// corpus's word count.
fn quote(text: &str, max: usize) -> String {
    let collapsed: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    truncate_with_ellipsis(&collapsed, max)
}

/// Render a markdown style-guide brief for `author` from their distilled
/// `patterns`, resolving each pattern's `example_ids` to verbatim quotes via
/// `samples_by_id`. `document_count`/`total_words` head the brief with the
/// corpus size the patterns were derived from.
pub fn render_style_guide(
    author: &str,
    patterns: &[VoicePattern],
    samples_by_id: &HashMap<Uuid, &VoiceSample>,
    document_count: usize,
    total_words: u32,
) -> String {
    let mut out = format!(
        "# Writing Voice: {author}\n\nDerived from {document_count} measured samples \
         ({total_words} words). Follow the Replicate patterns; recognize but never \
         reproduce anything under Avoid.\n"
    );

    for (heading, replicate) in [("Replicate", true), ("Avoid", false)] {
        let section: Vec<&VoicePattern> = patterns
            .iter()
            .filter(|p| p.replicate == replicate)
            .collect();
        if section.is_empty() {
            continue;
        }
        out.push_str(&format!("\n## {heading}\n"));
        for p in section {
            out.push_str(&format!(
                "\n### {}\n{}\n",
                category_heading(p.category),
                p.description
            ));
            let quotes: Vec<&str> = p
                .example_ids
                .iter()
                .filter_map(|id| samples_by_id.get(id))
                .map(|s| s.text.as_str())
                .take(MAX_QUOTES_PER_PATTERN)
                .collect();
            for q in quotes {
                out.push_str(&format!("- \"{}\"\n", quote(q, MAX_QUOTE_CHARS)));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{compute_linguistic_profile, word_count, Confidence, SourceKind, SourceRef};
    use chrono::Utc;

    fn test_sample(text: &str) -> VoiceSample {
        let profile = compute_linguistic_profile(text);
        VoiceSample {
            id: Uuid::new_v4(),
            author: "kevin".to_string(),
            text: text.to_string(),
            source: SourceRef {
                kind: SourceKind::Manual,
                project: None,
                session_id: None,
                locator: "test".to_string(),
            },
            captured_at: Utc::now(),
            word_count: word_count(text),
            register: None,
            tags: vec![],
            confidence: Confidence::Verbatim,
            profile,
        }
    }

    fn test_pattern(
        category: PatternCategory,
        description: &str,
        example_ids: Vec<Uuid>,
        replicate: bool,
    ) -> VoicePattern {
        VoicePattern {
            id: Uuid::new_v4(),
            author: "kevin".to_string(),
            category,
            description: description.to_string(),
            example_ids,
            replicate,
        }
    }

    #[test]
    fn category_heading_covers_every_variant_with_a_readable_label() {
        assert_eq!(category_heading(PatternCategory::Opener), "Opener");
        assert_eq!(
            category_heading(PatternCategory::SentenceConstruction),
            "Sentence Construction"
        );
        assert_eq!(category_heading(PatternCategory::ToneRule), "Tone");
        assert_eq!(category_heading(PatternCategory::Vocabulary), "Vocabulary");
        assert_eq!(
            category_heading(PatternCategory::AntiPattern),
            "Anti-Pattern"
        );
    }

    #[test]
    fn quote_collapses_whitespace_and_truncates_long_text() {
        assert_eq!(quote("one\ntwo   three", 100), "one two three");
        let long = "word ".repeat(50);
        let truncated = quote(&long, 10);
        assert_eq!(truncated.chars().count(), 11); // 10 chars + the ellipsis
        assert!(truncated.ends_with('…'));
    }

    #[test]
    fn quote_does_not_truncate_text_at_or_under_the_limit() {
        assert_eq!(quote("short", 100), "short");
    }

    #[test]
    fn render_style_guide_groups_by_replicate_not_by_category() {
        let s1 = test_sample("ok ship it");
        let replicate_pattern =
            test_pattern(PatternCategory::Opener, "Opens on ok", vec![s1.id], true);
        // An AntiPattern-categoried pattern that is still `replicate: true`
        // must land in Replicate, not Avoid — grouping is by the
        // `replicate` field, not by category naming.
        let odd_pattern = test_pattern(
            PatternCategory::AntiPattern,
            "Not actually an anti-pattern",
            vec![],
            true,
        );
        let avoid_pattern =
            test_pattern(PatternCategory::AntiPattern, "Never do this", vec![], false);

        let by_id: HashMap<Uuid, &VoiceSample> = [(s1.id, &s1)].into_iter().collect();
        let out = render_style_guide(
            "kevin",
            &[replicate_pattern, odd_pattern, avoid_pattern],
            &by_id,
            1,
            s1.word_count,
        );

        let replicate_idx = out.find("## Replicate").unwrap();
        let avoid_idx = out.find("## Avoid").unwrap();
        assert!(replicate_idx < avoid_idx);
        assert!(out[replicate_idx..avoid_idx].contains("Not actually an anti-pattern"));
        assert!(out[avoid_idx..].contains("Never do this"));
        assert!(!out[avoid_idx..].contains("Not actually an anti-pattern"));
    }

    #[test]
    fn render_style_guide_resolves_example_ids_to_verbatim_quotes() {
        let s1 = test_sample("ok ship the release");
        let pattern = test_pattern(PatternCategory::Opener, "Opens on ok", vec![s1.id], true);
        let by_id: HashMap<Uuid, &VoiceSample> = [(s1.id, &s1)].into_iter().collect();
        let out = render_style_guide("kevin", &[pattern], &by_id, 1, s1.word_count);
        assert!(out.contains("ok ship the release"));
    }

    #[test]
    fn render_style_guide_skips_dangling_example_ids_without_panicking() {
        let dangling_id = Uuid::new_v4();
        let pattern = test_pattern(
            PatternCategory::Opener,
            "Opens on ok",
            vec![dangling_id],
            true,
        );
        let by_id: HashMap<Uuid, &VoiceSample> = HashMap::new();
        // Must not panic despite the id resolving to nothing.
        let out = render_style_guide("kevin", &[pattern], &by_id, 0, 0);
        assert!(out.contains("Opens on ok"));
    }

    #[test]
    fn render_style_guide_caps_quotes_per_pattern() {
        let samples: Vec<VoiceSample> = (0..10)
            .map(|i| test_sample(&format!("ok ship number {i}")))
            .collect();
        let ids: Vec<Uuid> = samples.iter().map(|s| s.id).collect();
        let pattern = test_pattern(PatternCategory::Opener, "Opens on ok", ids, true);
        let by_id: HashMap<Uuid, &VoiceSample> = samples.iter().map(|s| (s.id, s)).collect();
        let total_words: u32 = samples.iter().map(|s| s.word_count).sum();
        let out = render_style_guide("kevin", &[pattern], &by_id, samples.len(), total_words);
        let quote_lines = out.lines().filter(|l| l.starts_with("- \"")).count();
        assert_eq!(quote_lines, MAX_QUOTES_PER_PATTERN);
    }
}
