//! `larc-lexicon-engine` — a schema and metrics engine for building
//! queryable writing-voice lexicons from real authored text.
//!
//! Four pieces:
//! - [`schema`] — `VoiceSample` (raw evidence) and `VoicePattern` (distilled rules).
//! - [`metrics`] — pure-Rust, dependency-light computation of `LinguisticProfile`.
//! - [`engine`] — the [`engine::LexiconEngine`] storage/retrieval trait.
//! - [`ingest`] — recovering genuinely human-authored text out of agent-session
//!   transcripts, which is harder than it looks and easy to get quietly wrong.
//!
//! The core carries no storage or embedding dependency. A reference backend
//! ([`backend::SqliteEngine`]) ships behind the off-by-default `sqlite-backend`
//! feature — see the crate README for why it is opt-in, and for how to
//! implement `LexiconEngine` against your own storage instead.

pub mod backend;
pub mod engine;
pub mod ingest;
pub mod metrics;
pub mod schema;

pub use engine::{LexiconEngine, LexiconError, Result};
pub use ingest::{extract_human_turns, strip_harness_blocks, ExtractedTurn};
pub use metrics::{compute_linguistic_profile, word_count};
pub use schema::{
    Confidence, LinguisticProfile, PatternCategory, Register, SourceKind, SourceRef, VoicePattern,
    VoiceSample,
};
