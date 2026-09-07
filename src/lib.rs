//! `larc-lexicon-engine` — a schema and metrics engine for building
//! queryable writing-voice lexicons from real authored text.
//!
//! Five pieces:
//! - [`schema`] — `VoiceSample` (raw evidence) and `VoicePattern` (distilled rules).
//! - [`metrics`] — pure-Rust, dependency-light computation of `LinguisticProfile`.
//! - [`engine`] — the [`engine::LexiconEngine`] storage/retrieval trait.
//! - [`ingest`] — recovering genuinely human-authored text out of agent-session
//!   transcripts, which is harder than it looks and easy to get quietly wrong.
//! - [`distill`] — deriving candidate `VoicePattern`s from measured evidence
//!   instead of writing them by hand.
//!
//! The core carries no storage or embedding dependency. A reference backend
//! ([`backend::SqliteEngine`]) ships behind the off-by-default `sqlite-backend`
//! feature — see the crate README for why it is opt-in, and for how to
//! implement `LexiconEngine` against your own storage instead.

pub mod backend;
pub mod distill;
pub mod engine;
pub mod ingest;
pub mod metrics;
pub mod schema;

pub use distill::distill_patterns;
pub use engine::{LexiconEngine, LexiconError, Result};
pub use ingest::{
    extract_human_turns, flag_pasted_content, strip_harness_blocks, ExtractedTurn, PasteSignal,
};
pub use metrics::{
    aggregate_corpus_profile, compute_linguistic_profile, contains_hedge, starts_with_imperative,
    word_count,
};
pub use schema::{
    Confidence, CorpusProfile, LinguisticProfile, PatternCategory, Register, SourceKind, SourceRef,
    VoicePattern, VoiceSample,
};

/// Deterministic id for an entity whose identity should come from its
/// content or origin rather than being randomly assigned — `UUID v5` over
/// `larc/<parts joined by "/">`. Shared by every id-deriving call site in
/// this crate (transcript-turn ids, file-ingest ids, distilled-pattern
/// ids) so the namespacing scheme itself — which `Uuid` namespace, what the
/// key looks like — exists in exactly one place. Deriving the same parts
/// twice always produces the same id, which is what lets a re-run ingest
/// or a re-run `distill_patterns` upsert instead of duplicate.
pub fn stable_uuid(parts: &[&str]) -> uuid::Uuid {
    let key = format!("larc/{}", parts.join("/"));
    uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_URL, key.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_uuid_is_deterministic_and_scoped_by_every_part() {
        assert_eq!(
            stable_uuid(&["distill", "kevin", "ack-opener"]),
            stable_uuid(&["distill", "kevin", "ack-opener"]),
        );
        assert_ne!(
            stable_uuid(&["distill", "kevin", "ack-opener"]),
            stable_uuid(&["distill", "kevin", "bare-imperative-opener"]),
        );
    }

    /// The refactor that introduced `stable_uuid` (extracted from two
    /// independent `Uuid::new_v5(&Uuid::NAMESPACE_URL, format!("larc/...")...)`
    /// call sites) must reproduce the exact ids either scheme already
    /// produced — a silent id shift here would break every already-stored
    /// sample's or pattern's upsert-on-re-run guarantee.
    #[test]
    fn stable_uuid_reproduces_the_original_pre_refactor_key_format() {
        let old_scheme = |key: &str| uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_URL, key.as_bytes());

        assert_eq!(
            stable_uuid(&["distill", "kevin", "ack-opener"]),
            old_scheme("larc/distill/kevin/ack-opener")
        );
        assert_eq!(
            stable_uuid(&["claude-session", "session-1#0", "yes"]),
            old_scheme("larc/claude-session/session-1#0/yes")
        );
    }
}
