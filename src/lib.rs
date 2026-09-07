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
