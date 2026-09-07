//! Deriving candidate [`VoicePattern`]s from measured evidence, rather than
//! writing them by hand.
//!
//! # Why this exists
//!
//! Every `VoicePattern` shipped in this crate's own dogfood session was
//! written by a human reading throwaway analysis scripts, then typing
//! `larc patterns add` by hand. That process has two real defects this
//! module fixes:
//!
//! 1. **No linked evidence.** A hand-written pattern's `example_ids` were
//!    always empty — the description cited a percentage, but nothing pointed
//!    back at which stored samples actually exhibited the trait.
//! 2. **Not reproducible.** Re-running the same analysis after the corpus
//!    grew meant re-deriving every number by hand and re-typing the CLI
//!    command, with no guarantee the new numbers were computed the same way
//!    as the old ones.
//!
//! [`distill_patterns`] computes a fixed set of discrete, per-turn markers by
//! scanning every sample's text individually — never by concatenating texts
//! first, which is the exact turn-fusion mistake [`crate::CorpusProfile`]
//! exists to avoid — plus one marker read directly from
//! [`crate::aggregate_corpus_profile`]'s pooled rates. Each marker becomes a
//! pattern only if it clears both a minimum sample size and a minimum effect
//! size; every emitted pattern's `description` cites the exact count and
//! percentage behind it, and `example_ids` names real stored samples that
//! exhibit the trait.
//!
//! Ids are deterministic per `(author, signal)` — UUID v5, not v4 — so
//! re-running `distill_patterns` as a corpus grows upserts each signal's
//! pattern with fresher numbers and evidence rather than accumulating
//! duplicates, matching [`crate::engine::LexiconEngine::save_patterns`]'s
//! `INSERT OR REPLACE` semantics.
//!
//! This deliberately does not attempt to derive anti-patterns (typos,
//! artifacts to recognize but never reproduce): distinguishing a genuine
//! typo from an unusual-but-intentional word choice needs a dictionary this
//! crate doesn't bundle, and a wrongly-flagged anti-pattern is a worse
//! failure than a missing one. That stays a manual `patterns add
//! --anti-pattern` call.

use uuid::Uuid;

use crate::metrics::{aggregate_corpus_profile, contains_hedge, starts_with_imperative};
use crate::schema::{PatternCategory, VoicePattern, VoiceSample};

/// Below this many samples, percentages are too noisy to make a claim from —
/// one or two turns can swing a rate by double digits. No pattern is emitted
/// at all until the corpus clears this bar.
const MIN_SAMPLES_TO_DISTILL: usize = 20;

/// A marker must clear this fraction of the corpus before it's considered a
/// real trait rather than something that merely happens sometimes.
const MIN_SHARE: f32 = 0.10;

/// Same idea as [`MIN_SHARE`] but for the two-sided causal/contrastive
/// connective comparison, which is a ratio rather than a share: the rarer
/// side must be outnumbered by at least this factor for the skew to be
/// worth naming.
const MIN_CONNECTIVE_RATIO: f32 = 3.0;

const ACK_OPENERS: &[&str] = &["ok", "okay", "yes", "yeah", "alright", "sure"];
const GRATITUDE_MARKERS: &[&str] = &["please", "thank", "thanks", "appreciate", "grateful"];
const MAX_EXAMPLE_IDS: usize = 3;

fn stable_pattern_id(author: &str, signal: &str) -> Uuid {
    Uuid::new_v5(
        &Uuid::NAMESPACE_URL,
        format!("larc/distill/{author}/{signal}").as_bytes(),
    )
}

fn first_word_lower(text: &str) -> Option<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '\'')
        .find(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
}

/// One discrete yes/no measurement over a corpus: how many samples hit, and
/// which ones, so a pattern built from it can cite real `example_ids`.
#[derive(Default)]
struct Signal {
    hits: usize,
    example_ids: Vec<Uuid>,
}

impl Signal {
    fn share(&self, total: usize) -> f32 {
        if total == 0 {
            0.0
        } else {
            self.hits as f32 / total as f32
        }
    }

    fn record(&mut self, id: Uuid) {
        self.hits += 1;
        if self.example_ids.len() < MAX_EXAMPLE_IDS {
            self.example_ids.push(id);
        }
    }
}

/// Derive candidate [`VoicePattern`]s from an author's stored samples.
///
/// Returns an empty `Vec` — not an error — when the corpus is smaller than
/// [`MIN_SAMPLES_TO_DISTILL`] or when no measured marker clears its
/// threshold. Silence here means "nothing measured strongly enough to
/// claim," never "nothing was checked."
pub fn distill_patterns(author: &str, samples: &[VoiceSample]) -> Vec<VoicePattern> {
    if samples.len() < MIN_SAMPLES_TO_DISTILL {
        return Vec::new();
    }
    let total = samples.len();

    let mut ack_opener = Signal::default();
    let mut bare_imperative = Signal::default();
    let mut no_terminal_punct = Signal::default();
    let mut gratitude = Signal::default();
    let mut hedges = Signal::default();

    for s in samples {
        let trimmed = s.text.trim();

        if let Some(first) = first_word_lower(trimmed) {
            if ACK_OPENERS.contains(&first.as_str()) {
                ack_opener.record(s.id);
            }
        }
        if starts_with_imperative(trimmed) {
            bare_imperative.record(s.id);
        }
        if !trimmed.is_empty() && !trimmed.ends_with(['.', '!', '?']) {
            no_terminal_punct.record(s.id);
        }
        let lower = trimmed.to_lowercase();
        if GRATITUDE_MARKERS.iter().any(|m| lower.contains(m)) {
            gratitude.record(s.id);
        }
        if contains_hedge(trimmed) {
            hedges.record(s.id);
        }
    }

    let mut patterns = Vec::new();

    if ack_opener.share(total) >= MIN_SHARE {
        patterns.push(VoicePattern {
            id: stable_pattern_id(author, "ack-opener"),
            author: author.to_string(),
            category: PatternCategory::Opener,
            description: format!(
                "Opens on a bare acknowledgment token before the actual content: {} of {} \
                 samples ({:.0}%) start with one of {ACK_OPENERS:?}.",
                ack_opener.hits,
                total,
                ack_opener.share(total) * 100.0
            ),
            example_ids: ack_opener.example_ids,
            replicate: true,
        });
    }

    if bare_imperative.share(total) >= MIN_SHARE {
        patterns.push(VoicePattern {
            id: stable_pattern_id(author, "bare-imperative-opener"),
            author: author.to_string(),
            category: PatternCategory::ToneRule,
            description: format!(
                "Directive-first: {} of {} samples ({:.0}%) open with a bare imperative verb, \
                 no softening clause.",
                bare_imperative.hits,
                total,
                bare_imperative.share(total) * 100.0
            ),
            example_ids: bare_imperative.example_ids,
            replicate: true,
        });
    }

    if no_terminal_punct.share(total) >= MIN_SHARE {
        patterns.push(VoicePattern {
            id: stable_pattern_id(author, "no-terminal-punctuation"),
            author: author.to_string(),
            category: PatternCategory::SentenceConstruction,
            description: format!(
                "Omits terminal punctuation: {} of {} samples ({:.0}%) end without a period, \
                 exclamation point, or question mark.",
                no_terminal_punct.hits,
                total,
                no_terminal_punct.share(total) * 100.0
            ),
            example_ids: no_terminal_punct.example_ids,
            replicate: true,
        });
    }

    if gratitude.share(total) >= MIN_SHARE {
        patterns.push(VoicePattern {
            id: stable_pattern_id(author, "gratitude-marker"),
            author: author.to_string(),
            category: PatternCategory::Vocabulary,
            description: format!(
                "Marks politeness with an explicit word rather than softening the request \
                 itself: {} of {} samples ({:.0}%) contain 'please', 'thank(s)', or \
                 'appreciate'.",
                gratitude.hits,
                total,
                gratitude.share(total) * 100.0
            ),
            example_ids: gratitude.example_ids,
            replicate: true,
        });
    }

    // Hedging is only worth naming when it's unusually *absent* — asserting
    // rather than qualifying is the notable trait; a middling hedge rate
    // isn't distinctive enough to claim either way.
    let hedge_share = hedges.share(total);
    if hedge_share > 0.0 && hedge_share < MIN_SHARE {
        patterns.push(VoicePattern {
            id: stable_pattern_id(author, "low-hedge-rate"),
            author: author.to_string(),
            category: PatternCategory::ToneRule,
            description: format!(
                "Asserts rather than hedges: only {} of {} samples ({:.0}%) contain any hedge \
                 word or phrase ('maybe', 'I think', 'probably', ...).",
                hedges.hits,
                total,
                hedge_share * 100.0
            ),
            example_ids: hedges.example_ids,
            replicate: true,
        });
    }

    // The one marker read from pooled corpus rates rather than a per-sample
    // scan — computed the same way `larc stats` computes every other rate.
    let texts: Vec<&str> = samples.iter().map(|s| s.text.as_str()).collect();
    let profile = aggregate_corpus_profile(texts);
    let causal = profile.causal_connective_rate_per_100_words;
    let contrast = profile.contrast_connective_rate_per_100_words;
    // Formulated as multiplication, not division, so a zero denominator
    // (the strongest possible skew — the other connective type never
    // appears at all) is naturally `true` rather than an edge case a
    // `contrast > 0.0` guard would wrongly exclude.
    let causal_dominant = causal > 0.0 && causal >= MIN_CONNECTIVE_RATIO * contrast;
    let contrast_dominant = contrast > 0.0 && contrast >= MIN_CONNECTIVE_RATIO * causal;
    if causal_dominant {
        let skew = if contrast > 0.0 {
            format!("a {:.0}:1 ratio", causal / contrast)
        } else {
            "no contrastive connective at all".to_string()
        };
        patterns.push(VoicePattern {
            id: stable_pattern_id(author, "causal-over-contrastive"),
            author: author.to_string(),
            category: PatternCategory::Vocabulary,
            description: format!(
                "Reasons in causal chains, rarely in contrasts: causal connectives run \
                 {causal:.2}/100w against {contrast:.2}/100w for contrastive ones — {skew}. \
                 Builds forward ('so', 'because') rather than pivoting corrective ('but', \
                 'however')."
            ),
            example_ids: Vec::new(),
            replicate: true,
        });
    } else if contrast_dominant {
        let skew = if causal > 0.0 {
            format!("a {:.0}:1 ratio", contrast / causal)
        } else {
            "no causal connective at all".to_string()
        };
        patterns.push(VoicePattern {
            id: stable_pattern_id(author, "contrastive-over-causal"),
            author: author.to_string(),
            category: PatternCategory::Vocabulary,
            description: format!(
                "Qualifies and concedes more than it chains reasons forward: contrastive \
                 connectives run {contrast:.2}/100w against {causal:.2}/100w for causal \
                 ones — {skew}."
            ),
            example_ids: Vec::new(),
            replicate: true,
        });
    }

    patterns
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::compute_linguistic_profile;
    use crate::schema::{Confidence, SourceKind, SourceRef};
    use chrono::Utc;

    fn sample(text: &str) -> VoiceSample {
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
            word_count: crate::metrics::word_count(text),
            register: None,
            tags: vec![],
            confidence: Confidence::Verbatim,
            profile: compute_linguistic_profile(text),
        }
    }

    fn corpus(n: usize, mut make: impl FnMut(usize) -> String) -> Vec<VoiceSample> {
        (0..n).map(|i| sample(&make(i))).collect()
    }

    #[test]
    fn below_minimum_sample_size_yields_nothing() {
        let samples = corpus(5, |i| format!("ok go ahead with number {i}"));
        assert!(distill_patterns("kevin", &samples).is_empty());
    }

    #[test]
    fn ack_opener_pattern_fires_with_evidence_when_it_clears_threshold() {
        // 15 of 25 open on an ack token -- well over MIN_SHARE.
        let mut samples = corpus(15, |i| format!("ok ship the change number {i}"));
        samples.extend(corpus(10, |i| {
            format!("investigate the failure in module {i}")
        }));

        let patterns = distill_patterns("kevin", &samples);
        let ack = patterns
            .iter()
            .find(|p| p.category == PatternCategory::Opener)
            .expect("ack-opener pattern should fire");
        assert!(ack.description.contains("15 of 25"));
        assert!(!ack.example_ids.is_empty(), "must cite real evidence");
        assert!(ack.example_ids.len() <= 3);
    }

    #[test]
    fn signal_below_threshold_does_not_fire() {
        // Only 1 of 25 -- 4%, under MIN_SHARE.
        let mut samples = corpus(1, |_| "ok go".to_string());
        samples.extend(corpus(24, |i| format!("investigate module {i} thoroughly")));
        let patterns = distill_patterns("kevin", &samples);
        assert!(!patterns
            .iter()
            .any(|p| p.category == PatternCategory::Opener));
    }

    #[test]
    fn no_terminal_punctuation_pattern_measures_correctly() {
        let mut samples = corpus(20, |i| format!("ship it number {i}"));
        samples.extend(corpus(5, |i| format!("Ship it, number {i}.")));

        let patterns = distill_patterns("kevin", &samples);
        let p = patterns
            .iter()
            .find(|p| p.category == PatternCategory::SentenceConstruction)
            .expect("no-terminal-punctuation pattern should fire");
        assert!(p.description.contains("20 of 25"));
    }

    #[test]
    fn low_hedge_rate_fires_only_when_hedging_is_rare_not_absent_or_common() {
        // Zero hedges at all -- share is exactly 0.0, deliberately excluded:
        // "never observed" is a weaker claim than "measured as rare."
        let none = corpus(25, |i| format!("ship the release number {i}"));
        assert!(!distill_patterns("kevin", &none)
            .iter()
            .any(|p| p.description.contains("hedge")));

        // 2 of 25 (8%) hedge -- rare but present -- should fire.
        let mut some = corpus(2, |i| format!("maybe we should ship number {i}"));
        some.extend(corpus(23, |i| format!("ship the release number {i}")));
        assert!(distill_patterns("kevin", &some)
            .iter()
            .any(|p| p.description.contains("hedge")));
    }

    #[test]
    fn causal_over_contrastive_reads_from_pooled_corpus_rates() {
        let mut samples = corpus(20, |i| format!("do it because it works well number {i}"));
        samples.extend(corpus(5, |i| format!("plain statement number {i}")));

        let patterns = distill_patterns("kevin", &samples);
        assert!(patterns
            .iter()
            .any(|p| p.description.contains("causal chains")));
    }

    /// Zero contrastive connectives at all is the *strongest* possible skew
    /// toward causal reasoning, not an edge case to exclude. A naive
    /// `causal / contrast >= THRESHOLD` guarded by `contrast > 0.0` would
    /// wrongly skip exactly this corpus.
    #[test]
    fn zero_contrastive_connectives_is_the_strongest_causal_skew_not_excluded() {
        let mut samples = corpus(20, |i| format!("do it because it works well number {i}"));
        samples.extend(corpus(5, |i| format!("plain statement number {i}")));

        let patterns = distill_patterns("kevin", &samples);
        let p = patterns
            .iter()
            .find(|p| p.description.contains("causal chains"))
            .expect("zero-denominator causal skew must still fire");
        assert!(p.description.contains("no contrastive connective at all"));
    }

    #[test]
    fn pattern_ids_are_deterministic_across_runs() {
        let samples = corpus(25, |i| format!("ok ship it number {i}"));
        let first = distill_patterns("kevin", &samples);
        let second = distill_patterns("kevin", &samples);
        assert_eq!(first.len(), second.len());
        for (a, b) in first.iter().zip(second.iter()) {
            assert_eq!(a.id, b.id, "same signal must produce the same id every run");
        }
    }

    #[test]
    fn different_authors_get_different_ids_for_the_same_signal() {
        let samples = corpus(25, |i| format!("ok ship it number {i}"));
        let kevin = distill_patterns("kevin", &samples);
        let other = distill_patterns("someone-else", &samples);
        assert_eq!(kevin[0].category, other[0].category);
        assert_ne!(kevin[0].id, other[0].id);
    }
}
