#![cfg(feature = "sqlite-backend")]

use chrono::Utc;
use larc_lexicon_engine::backend::SqliteEngine;
use larc_lexicon_engine::{
    aggregate_corpus_profile, compute_linguistic_profile, word_count, Confidence, LexiconEngine,
    PatternCategory, SourceKind, SourceRef, VoicePattern, VoiceSample,
};
use uuid::Uuid;

fn sample_with_id(id: Uuid, author: &str, text: &str) -> VoiceSample {
    let profile = compute_linguistic_profile(text);
    VoiceSample {
        id,
        author: author.to_string(),
        text: text.to_string(),
        source: SourceRef {
            kind: SourceKind::Manual,
            project: None,
            session_id: None,
            locator: "test".to_string(),
        },
        captured_at: Utc::now(),
        // Uses the crate's canonical tokenizer rather than `split_whitespace`,
        // so this count is denominated identically to the `*_per_100_words`
        // rates in the profile above it.
        word_count: word_count(text),
        register: None,
        tags: vec![],
        confidence: Confidence::Verbatim,
        profile,
    }
}

fn sample(author: &str, text: &str) -> VoiceSample {
    sample_with_id(Uuid::new_v4(), author, text)
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

/// Re-ingesting the same source must not duplicate it.
///
/// The CLI derives sample ids deterministically (UUID v5 over origin + exact
/// text) precisely so that ingest commands can be re-run as a corpus grows.
/// That guarantee only holds if the backend upserts on `id` rather than
/// appending, so it is asserted here rather than assumed.
#[tokio::test]
async fn reingesting_the_same_id_replaces_rather_than_duplicates() {
    let dir = tempdir();
    let db_path = dir.join("lexicon.sqlite3");
    let engine = SqliteEngine::open(&db_path).expect("open sqlite engine");

    let stable = Uuid::new_v5(&Uuid::NAMESPACE_URL, b"larc/test/idempotency");
    let first = sample_with_id(
        stable,
        "kevin",
        "Check the deploy logs before pushing to main.",
    );
    engine.ingest(&[first]).await.expect("first ingest");
    engine
        .ingest(&[sample_with_id(
            stable,
            "kevin",
            "Check the deploy logs before pushing to main.",
        )])
        .await
        .expect("second ingest");

    let all = engine
        .search("deploy logs", Some("kevin"), 50)
        .await
        .expect("search");
    assert_eq!(all.len(), 1, "identical re-ingest must not duplicate rows");

    std::fs::remove_file(&db_path).ok();
}

/// `patterns_for` can only ever return what `save_patterns` wrote — before the
/// trait carried a write method the read side was structurally dead, so the
/// round trip is asserted end to end.
#[tokio::test]
async fn patterns_save_and_read_back_round_trip() {
    let dir = tempdir();
    let db_path = dir.join("lexicon.sqlite3");
    let engine = SqliteEngine::open(&db_path).expect("open sqlite engine");

    let example = Uuid::new_v4();
    let pattern = VoicePattern {
        id: Uuid::new_v4(),
        author: "kevin".to_string(),
        category: PatternCategory::AntiPattern,
        description: "Drops the article in hurried directives — recognize, never reproduce."
            .to_string(),
        example_ids: vec![example],
        replicate: false,
    };

    let written = engine
        .save_patterns(std::slice::from_ref(&pattern))
        .await
        .expect("save patterns");
    assert_eq!(written, 1);

    let read_back = engine.patterns_for("kevin").await.expect("patterns_for");
    assert_eq!(read_back.len(), 1);
    assert_eq!(read_back[0].id, pattern.id);
    assert_eq!(read_back[0].description, pattern.description);
    assert_eq!(read_back[0].example_ids, vec![example]);
    assert!(
        !read_back[0].replicate,
        "anti-pattern must survive the round trip as non-replicable"
    );

    assert!(
        engine
            .patterns_for("someone-else")
            .await
            .expect("patterns_for")
            .is_empty(),
        "patterns must be scoped to their author"
    );

    std::fs::remove_file(&db_path).ok();
}

/// `samples_for` is a listing, not a ranked `search` — it must return every
/// stored sample for an author, never a `top_k`-limited subset, since
/// `aggregate_corpus_profile` (what `larc stats` calls it for) needs the
/// whole population to pool correctly.
#[tokio::test]
async fn samples_for_returns_the_whole_population_not_a_ranked_subset() {
    let dir = tempdir();
    let db_path = dir.join("lexicon.sqlite3");
    let engine = SqliteEngine::open(&db_path).expect("open sqlite engine");

    let kevin_samples: Vec<VoiceSample> = (0..5)
        .map(|i| sample("kevin", &format!("This is sample number {i} for kevin.")))
        .collect();
    engine.ingest(&kevin_samples).await.expect("ingest kevin");
    engine
        .ingest(&[sample("someone-else", "A different author's sample.")])
        .await
        .expect("ingest other author");

    let kevin_all = engine
        .samples_for(Some("kevin"))
        .await
        .expect("samples_for kevin");
    assert_eq!(
        kevin_all.len(),
        5,
        "samples_for must return every sample, not a top_k-limited subset"
    );

    let everyone = engine.samples_for(None).await.expect("samples_for all");
    assert_eq!(everyone.len(), 6, "None must return the whole lexicon");

    // Integration check: the actual pipeline `larc stats` runs.
    let texts: Vec<&str> = kevin_all.iter().map(|s| s.text.as_str()).collect();
    let profile = aggregate_corpus_profile(texts);
    assert_eq!(profile.document_count, 5);

    std::fs::remove_file(&db_path).ok();
}

fn tempdir() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("larc-lexicon-test-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}
