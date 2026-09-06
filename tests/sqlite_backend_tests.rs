#![cfg(feature = "sqlite-backend")]

use chrono::Utc;
use larc_lexicon_engine::backend::SqliteEngine;
use larc_lexicon_engine::{
    compute_linguistic_profile, Confidence, LexiconEngine, SourceKind, SourceRef, VoiceSample,
};
use uuid::Uuid;

fn sample(author: &str, text: &str) -> VoiceSample {
    let profile = compute_linguistic_profile(text);
    VoiceSample {
        id: Uuid::new_v4(),
        author: author.to_string(),
        text: text.to_string(),
        source: SourceRef {
            kind: SourceKind::Manual,
            project: None,
            session_id: None,
            locator: "test".to_string(),
        },
        captured_at: Utc::now(),
        word_count: text.split_whitespace().count() as u32,
        register: None,
        tags: vec![],
        confidence: Confidence::Verbatim,
        profile,
    }
}

#[tokio::test]
async fn ingest_and_search_round_trip() {
    let dir = tempdir();
    let db_path = dir.join("lexicon.sqlite3");
    let engine = SqliteEngine::open(&db_path).expect("open sqlite engine");

    let samples = vec![
        sample("kevin", "Check the deploy logs before pushing to main."),
        sample(
            "kevin",
            "I think the readability formula might be off by a bit.",
        ),
        sample("kevin", "The quarterly numbers show a clear upward trend."),
    ];

    let ingested = engine.ingest(&samples).await.expect("ingest");
    assert_eq!(ingested, 3);

    let results = engine
        .search("run the deployment checks", Some("kevin"), 1)
        .await
        .expect("search");
    assert_eq!(results.len(), 1);
    assert!(results[0].text.contains("deploy"));

    std::fs::remove_file(&db_path).ok();
}

fn tempdir() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("larc-lexicon-test-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}
