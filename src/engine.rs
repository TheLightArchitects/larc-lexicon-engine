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
    async fn patterns_for(&self, author: &str) -> Result<Vec<VoicePattern>>;
}
