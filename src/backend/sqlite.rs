use std::path::Path;
use std::sync::Mutex;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use fastembed::{TextEmbedding, TextInitOptions};
use rusqlite::{params, Connection};
use uuid::Uuid;

use crate::engine::{LexiconEngine, LexiconError, Result};
use crate::schema::{
    Confidence, LinguisticProfile, PatternCategory, Register, SourceRef, VoicePattern, VoiceSample,
};

const SCHEMA_SQL: &str = "
CREATE TABLE IF NOT EXISTS samples (
    id TEXT PRIMARY KEY,
    author TEXT NOT NULL,
    text TEXT NOT NULL,
    source_json TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    word_count INTEGER NOT NULL,
    register_json TEXT NOT NULL,
    tags_json TEXT NOT NULL,
    confidence_json TEXT NOT NULL,
    profile_json TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_samples_author ON samples(author);

CREATE TABLE IF NOT EXISTS patterns (
    id TEXT PRIMARY KEY,
    author TEXT NOT NULL,
    category_json TEXT NOT NULL,
    description TEXT NOT NULL,
    example_ids_json TEXT NOT NULL,
    replicate INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_patterns_author ON patterns(author);
";

fn embedding_to_blob(embedding: &[f32]) -> Vec<u8> {
    embedding.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn blob_to_embedding(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// A SQLite-backed `LexiconEngine` using local `fastembed` embeddings and
/// brute-force cosine similarity search (`fastembed::similarity::top_k`) --
/// entirely local, no external service, no network after the first model
/// download.
pub struct SqliteEngine {
    conn: Mutex<Connection>,
    /// `None` until the first call that actually needs to embed something.
    /// `patterns_for`/`save_patterns`/`samples_for` never touch this, so a
    /// command that only reads or writes patterns, or lists samples, never
    /// pays the ONNX model's load cost — a real latency win for `larc
    /// stats`/`distill`/`style-guide`/`patterns *`, which have no use for
    /// an embedding model at all.
    model: Mutex<Option<TextEmbedding>>,
}

impl SqliteEngine {
    pub fn open(db_path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(db_path).map_err(|e| LexiconError::Storage(e.to_string()))?;
        conn.execute_batch(SCHEMA_SQL)
            .map_err(|e| LexiconError::Storage(e.to_string()))?;
        Ok(Self {
            conn: Mutex::new(conn),
            model: Mutex::new(None),
        })
    }

    /// Lock the model slot, initializing it on first use, and return the
    /// guard so a caller can embed through it without a second lock/unlock
    /// round trip.
    fn ensure_model(&self) -> Result<std::sync::MutexGuard<'_, Option<TextEmbedding>>> {
        let mut guard = self
            .model
            .lock()
            .map_err(|_| LexiconError::Embedding("embedding model lock poisoned".into()))?;
        if guard.is_none() {
            *guard = Some(
                TextEmbedding::try_new(TextInitOptions::default())
                    .map_err(|e| LexiconError::Embedding(e.to_string()))?,
            );
        }
        Ok(guard)
    }

    #[cfg(test)]
    fn model_is_loaded(&self) -> bool {
        self.model.lock().map(|g| g.is_some()).unwrap_or(false)
    }

    fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let mut guard = self.ensure_model()?;
        let model = guard.as_mut().ok_or_else(|| {
            LexiconError::Embedding("embedding model unexpectedly absent after init".into())
        })?;
        let mut embeddings = model
            .embed(vec![text], None)
            .map_err(|e| LexiconError::Embedding(e.to_string()))?;
        embeddings
            .pop()
            .ok_or_else(|| LexiconError::Embedding("fastembed returned no embedding".into()))
    }

    /// Every stored sample (optionally scoped to one author) with its
    /// embedding — the shared substrate `search` ranks and `samples_for`
    /// returns unranked. Kept as one query so the two can never drift onto
    /// different column lists or filters.
    fn select_samples_with_embedding(
        &self,
        author: Option<&str>,
    ) -> Result<Vec<(VoiceSample, Vec<f32>)>> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| LexiconError::Storage("connection lock poisoned".into()))?;

        const SELECT: &str = "SELECT id, author, text, source_json, captured_at, word_count, register_json, tags_json, confidence_json, profile_json, embedding FROM samples";
        let mut rows: Vec<(VoiceSample, Vec<f32>)> = Vec::new();
        if let Some(author) = author {
            let mut stmt = conn
                .prepare(&format!("{SELECT} WHERE author = ?1"))
                .map_err(|e| LexiconError::Storage(e.to_string()))?;
            let mapped = stmt
                .query_map(params![author], row_to_sample_with_embedding)
                .map_err(|e| LexiconError::Storage(e.to_string()))?;
            for row in mapped {
                rows.push(row.map_err(|e| LexiconError::Storage(e.to_string()))?);
            }
        } else {
            let mut stmt = conn
                .prepare(SELECT)
                .map_err(|e| LexiconError::Storage(e.to_string()))?;
            let mapped = stmt
                .query_map([], row_to_sample_with_embedding)
                .map_err(|e| LexiconError::Storage(e.to_string()))?;
            for row in mapped {
                rows.push(row.map_err(|e| LexiconError::Storage(e.to_string()))?);
            }
        }
        Ok(rows)
    }
}

#[async_trait]
impl LexiconEngine for SqliteEngine {
    async fn ingest(&self, samples: &[VoiceSample]) -> Result<usize> {
        if samples.is_empty() {
            return Ok(0);
        }
        let texts: Vec<&str> = samples.iter().map(|s| s.text.as_str()).collect();
        let embeddings = {
            let mut guard = self.ensure_model()?;
            let model = guard.as_mut().ok_or_else(|| {
                LexiconError::Embedding("embedding model unexpectedly absent after init".into())
            })?;
            model
                .embed(texts, None)
                .map_err(|e| LexiconError::Embedding(e.to_string()))?
        };

        let conn = self
            .conn
            .lock()
            .map_err(|_| LexiconError::Storage("connection lock poisoned".into()))?;
        for (sample, embedding) in samples.iter().zip(embeddings.iter()) {
            conn.execute(
                "INSERT OR REPLACE INTO samples
                 (id, author, text, source_json, captured_at, word_count, register_json, tags_json, confidence_json, profile_json, embedding)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                params![
                    sample.id.to_string(),
                    sample.author,
                    sample.text,
                    serde_json::to_string(&sample.source).map_err(|e| LexiconError::Storage(e.to_string()))?,
                    sample.captured_at.to_rfc3339(),
                    sample.word_count,
                    serde_json::to_string(&sample.register).map_err(|e| LexiconError::Storage(e.to_string()))?,
                    serde_json::to_string(&sample.tags).map_err(|e| LexiconError::Storage(e.to_string()))?,
                    serde_json::to_string(&sample.confidence).map_err(|e| LexiconError::Storage(e.to_string()))?,
                    serde_json::to_string(&sample.profile).map_err(|e| LexiconError::Storage(e.to_string()))?,
                    embedding_to_blob(embedding),
                ],
            )
            .map_err(|e| LexiconError::Storage(e.to_string()))?;
        }
        Ok(samples.len())
    }

    async fn search(
        &self,
        query: &str,
        author: Option<&str>,
        top_k: usize,
    ) -> Result<Vec<VoiceSample>> {
        let query_embedding = self.embed_one(query)?;
        let rows = self.select_samples_with_embedding(author)?;

        let corpus: Vec<&[f32]> = rows.iter().map(|(_, e)| e.as_slice()).collect();
        let ranked = fastembed::similarity::top_k(&query_embedding, &corpus, top_k);

        Ok(ranked
            .into_iter()
            .map(|(idx, _score)| rows[idx].0.clone())
            .collect())
    }

    async fn samples_for(&self, author: Option<&str>) -> Result<Vec<VoiceSample>> {
        Ok(self
            .select_samples_with_embedding(author)?
            .into_iter()
            .map(|(sample, _embedding)| sample)
            .collect())
    }

    async fn patterns_for(&self, author: &str) -> Result<Vec<VoicePattern>> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| LexiconError::Storage("connection lock poisoned".into()))?;
        let mut stmt = conn
            .prepare("SELECT id, author, category_json, description, example_ids_json, replicate FROM patterns WHERE author = ?1")
            .map_err(|e| LexiconError::Storage(e.to_string()))?;
        let mapped = stmt
            .query_map(params![author], row_to_pattern)
            .map_err(|e| LexiconError::Storage(e.to_string()))?;
        let mut out = Vec::new();
        for row in mapped {
            out.push(row.map_err(|e| LexiconError::Storage(e.to_string()))?);
        }
        Ok(out)
    }

    async fn save_patterns(&self, patterns: &[VoicePattern]) -> Result<usize> {
        if patterns.is_empty() {
            return Ok(0);
        }
        let conn = self
            .conn
            .lock()
            .map_err(|_| LexiconError::Storage("connection lock poisoned".into()))?;
        for pattern in patterns {
            conn.execute(
                "INSERT OR REPLACE INTO patterns
                 (id, author, category_json, description, example_ids_json, replicate)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    pattern.id.to_string(),
                    pattern.author,
                    serde_json::to_string(&pattern.category)
                        .map_err(|e| LexiconError::Storage(e.to_string()))?,
                    pattern.description,
                    serde_json::to_string(&pattern.example_ids)
                        .map_err(|e| LexiconError::Storage(e.to_string()))?,
                    pattern.replicate,
                ],
            )
            .map_err(|e| LexiconError::Storage(e.to_string()))?;
        }
        Ok(patterns.len())
    }

    async fn delete_patterns(&self, ids: &[Uuid]) -> Result<usize> {
        if ids.is_empty() {
            return Ok(0);
        }
        let conn = self
            .conn
            .lock()
            .map_err(|_| LexiconError::Storage("connection lock poisoned".into()))?;
        let mut deleted = 0usize;
        for id in ids {
            deleted += conn
                .execute(
                    "DELETE FROM patterns WHERE id = ?1",
                    params![id.to_string()],
                )
                .map_err(|e| LexiconError::Storage(e.to_string()))?;
        }
        Ok(deleted)
    }
}

fn row_to_sample_with_embedding(row: &rusqlite::Row) -> rusqlite::Result<(VoiceSample, Vec<f32>)> {
    use rusqlite::types::Type;

    let id_str: String = row.get(0)?;
    let author: String = row.get(1)?;
    let text: String = row.get(2)?;
    let source_json: String = row.get(3)?;
    let captured_at_str: String = row.get(4)?;
    let word_count: u32 = row.get(5)?;
    let register_json: String = row.get(6)?;
    let tags_json: String = row.get(7)?;
    let confidence_json: String = row.get(8)?;
    let profile_json: String = row.get(9)?;
    let embedding_blob: Vec<u8> = row.get(10)?;

    let id = Uuid::parse_str(&id_str)
        .map_err(|e| rusqlite::Error::FromSqlConversionFailure(0, Type::Text, Box::new(e)))?;
    let source: SourceRef = serde_json::from_str(&source_json)
        .map_err(|e| rusqlite::Error::FromSqlConversionFailure(3, Type::Text, Box::new(e)))?;
    let captured_at = DateTime::parse_from_rfc3339(&captured_at_str)
        .map_err(|e| rusqlite::Error::FromSqlConversionFailure(4, Type::Text, Box::new(e)))?
        .with_timezone(&Utc);
    let register: Option<Register> = serde_json::from_str(&register_json)
        .map_err(|e| rusqlite::Error::FromSqlConversionFailure(6, Type::Text, Box::new(e)))?;
    let tags: Vec<String> = serde_json::from_str(&tags_json)
        .map_err(|e| rusqlite::Error::FromSqlConversionFailure(7, Type::Text, Box::new(e)))?;
    let confidence: Confidence = serde_json::from_str(&confidence_json)
        .map_err(|e| rusqlite::Error::FromSqlConversionFailure(8, Type::Text, Box::new(e)))?;
    let profile: LinguisticProfile = serde_json::from_str(&profile_json)
        .map_err(|e| rusqlite::Error::FromSqlConversionFailure(9, Type::Text, Box::new(e)))?;

    let embedding = blob_to_embedding(&embedding_blob);

    Ok((
        VoiceSample {
            id,
            author,
            text,
            source,
            captured_at,
            word_count,
            register,
            tags,
            confidence,
            profile,
        },
        embedding,
    ))
}

fn row_to_pattern(row: &rusqlite::Row) -> rusqlite::Result<VoicePattern> {
    use rusqlite::types::Type;

    let id_str: String = row.get(0)?;
    let author: String = row.get(1)?;
    let category_json: String = row.get(2)?;
    let description: String = row.get(3)?;
    let example_ids_json: String = row.get(4)?;
    let replicate: bool = row.get(5)?;

    let id = Uuid::parse_str(&id_str)
        .map_err(|e| rusqlite::Error::FromSqlConversionFailure(0, Type::Text, Box::new(e)))?;
    let category: PatternCategory = serde_json::from_str(&category_json)
        .map_err(|e| rusqlite::Error::FromSqlConversionFailure(2, Type::Text, Box::new(e)))?;
    let example_ids: Vec<Uuid> = serde_json::from_str(&example_ids_json)
        .map_err(|e| rusqlite::Error::FromSqlConversionFailure(4, Type::Text, Box::new(e)))?;

    Ok(VoicePattern {
        id,
        author,
        category,
        description,
        example_ids,
        replicate,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tempdir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("larc-lexicon-lazy-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    /// `open` must not pay the ONNX model's load cost for callers that
    /// never embed anything — `patterns_for`/`save_patterns`/`samples_for`
    /// have no use for it, and this is what makes `larc patterns`/`larc
    /// stats`/`larc distill` fast regardless of how large the embedding
    /// model is.
    #[tokio::test]
    async fn open_does_not_eagerly_load_the_embedding_model() {
        let dir = tempdir();
        let db_path = dir.join("lexicon.sqlite3");
        let engine = SqliteEngine::open(&db_path).expect("open sqlite engine");
        assert!(
            !engine.model_is_loaded(),
            "open() must not initialize the embedding model"
        );

        // Patterns-only operations must not trigger it either.
        let pattern = VoicePattern {
            id: Uuid::new_v4(),
            author: "kevin".to_string(),
            category: PatternCategory::ToneRule,
            description: "test".to_string(),
            example_ids: vec![],
            replicate: true,
        };
        engine
            .save_patterns(&[pattern])
            .await
            .expect("save patterns");
        engine.patterns_for("kevin").await.expect("patterns_for");
        assert!(
            !engine.model_is_loaded(),
            "reading/writing patterns must not load the embedding model"
        );

        std::fs::remove_file(&db_path).ok();
    }
}
