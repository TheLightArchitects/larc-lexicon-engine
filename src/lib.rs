//! `larc-lexicon-engine` — a schema and metrics engine for building
//! queryable writing-voice lexicons from real authored text.
//!
//! Three pieces:
//! - [`schema`] — `VoiceSample` (raw evidence) and `VoicePattern` (distilled rules).
//! - [`metrics`] — pure-Rust, dependency-light computation of `LinguisticProfile`.
//! - [`engine`] — the [`engine::LexiconEngine`] storage/retrieval trait.
//!
//! No backend implementation ships in this crate by design — see the crate
//! README for why, and how to implement `LexiconEngine` against your own
//! storage.

pub mod engine;
pub mod metrics;
pub mod schema;

pub use engine::{LexiconEngine, LexiconError, Result};
pub use metrics::compute_linguistic_profile;
pub use schema::{
    Confidence, LinguisticProfile, PatternCategory, Register, SourceKind, SourceRef, VoicePattern,
    VoiceSample,
};
