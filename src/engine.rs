use async_trait::async_trait;
use thiserror::Error;

use crate::schema::{VoicePattern, VoiceSample};

#[derive(Debug, Error)]
pub enum LexiconError {
    #[error("storage error: {0}")]
    Storage(String),
    #[error("embedding error: {0}")]
    Embedding(String),
    #[error("not found")]
    NotFound,
}

pub type Result<T> = std::result::Result<T, LexiconError>;

/// Retrieval/storage abstraction. Deliberately a trait, not a hard dependency
/// on any one backend — a SQLite+embedding implementation is the obvious
/// first backend, but nothing in this crate assumes it. A consumer can
/// implement this trait against their own storage (e.g. a private,
/// already-deployed vector store) without this crate ever depending on it.
#[async_trait]
pub trait LexiconEngine {
    async fn ingest(&self, samples: &[VoiceSample]) -> Result<usize>;
    async fn search(
        &self,
        query: &str,
        author: Option<&str>,
        top_k: usize,
    ) -> Result<Vec<VoiceSample>>;
    /// Every stored sample for an author (or the whole lexicon if `None`),
    /// in no particular order.
    ///
    /// Deliberately separate from [`LexiconEngine::search`], which is a
    /// `top_k` embedding-similarity *ranking* — the wrong tool for anything
    /// that needs the whole population, such as
    /// [`crate::metrics::aggregate_corpus_profile`]. There is no `top_k`
    /// value that means "all of them, unranked," and running a similarity
    /// search at all costs an embedding computation this method has no use
    /// for.
    async fn samples_for(&self, author: Option<&str>) -> Result<Vec<VoiceSample>>;
    async fn patterns_for(&self, author: &str) -> Result<Vec<VoicePattern>>;
    /// Persist distilled patterns, replacing any existing pattern with the
    /// same id. Returns how many were written.
    ///
    /// Kept separate from [`LexiconEngine::ingest`] because the two layers are
    /// written under different circumstances: samples are bulk-loaded raw
    /// evidence, whereas a `VoicePattern` is a reviewed, human-legible claim
    /// *about* that evidence. Without this method `patterns_for` could only
    /// ever return empty, since nothing else in the trait can populate it.
    async fn save_patterns(&self, patterns: &[VoicePattern]) -> Result<usize>;
}
