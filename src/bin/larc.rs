//! `larc` — command-line interface to the lexicon engine.
//!
//! Subcommands split by what they actually need. `profile` requires nothing
//! but the core crate, so it is always present. Everything that reads or
//! writes a lexicon needs storage and embeddings and is therefore compiled
//! only with the `sqlite-backend` feature — rather than shipping commands that
//! exist in `--help` but fail at runtime.

use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};
use larc_lexicon_engine::{compute_linguistic_profile, LinguisticProfile};

#[derive(Parser)]
#[command(
    name = "larc",
    version,
    about = "Build and query writing-voice lexicons from real authored text."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compute the linguistic profile of a text file, without storing anything.
    Profile(ProfileCmd),
    /// Add authored text to a lexicon.
    #[cfg(feature = "sqlite-backend")]
    Ingest(IngestCmd),
    /// Semantic search over stored samples.
    #[cfg(feature = "sqlite-backend")]
    Search(SearchCmd),
    /// Read and write distilled voice patterns.
    #[cfg(feature = "sqlite-backend")]
    Patterns(PatternsCmd),
}

// ---------------------------------------------------------------------------
// profile
// ---------------------------------------------------------------------------

#[derive(Args)]
struct ProfileCmd {
    /// Text file to profile. Use `-` to read stdin.
    path: String,
    /// Emit the raw `LinguisticProfile` as JSON.
    #[arg(long)]
    json: bool,
}

fn read_input(path: &str) -> Result<String, String> {
    if path == "-" {
        use std::io::Read;
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .map_err(|e| format!("reading stdin: {e}"))?;
        Ok(buf)
    } else {
        std::fs::read_to_string(path).map_err(|e| format!("reading {path}: {e}"))
    }
}

/// Render a deferred metric. These are `None` by design, not by omission —
/// printing `0.0` would present an unmeasured field as a measurement.
fn deferred(value: Option<f32>, needs: &str) -> String {
    match value {
        Some(v) => format!("{v:.3}"),
        None => format!("— (not measured: needs {needs})"),
    }
}

fn render_profile(p: &LinguisticProfile, words: u32) -> String {
    format!(
        "Words: {words}

Lexical richness
  type-token ratio            {:.3}
  avg word length             {:.2}
  hapax legomena ratio        {:.3}

Syntactic rhythm
  avg sentence length         {:.2}
  sentence length stddev      {:.2}
  comma      /100w            {:.2}
  semicolon  /100w            {:.2}
  dash       /100w            {:.2}
  question   /100w            {:.2}
  exclamation/100w            {:.2}

Readability
  Flesch reading ease         {:.1}
  Flesch-Kincaid grade        {:.1}
  Gunning fog index           {:.1}

Person & address
  1st person singular /100w   {:.2}
  1st person plural   /100w   {:.2}
  2nd person          /100w   {:.2}
  imperative rate             {:.3}

Epistemic stance
  certainty /100w             {:.2}
  hedge     /100w             {:.2}

Cognitive connectives
  causal    /100w             {:.2}
  contrast  /100w             {:.2}

Deferred
  formality score             {}
  sentiment polarity          {}
  sentiment subjectivity      {}",
        p.type_token_ratio,
        p.avg_word_length,
        p.hapax_legomena_ratio,
        p.avg_sentence_length,
        p.sentence_length_stddev,
        p.comma_rate_per_100_words,
        p.semicolon_rate_per_100_words,
        p.dash_rate_per_100_words,
        p.question_rate_per_100_words,
        p.exclamation_rate_per_100_words,
        p.flesch_reading_ease,
        p.flesch_kincaid_grade,
        p.gunning_fog_index,
        p.first_person_singular_rate_per_100_words,
        p.first_person_plural_rate_per_100_words,
        p.second_person_rate_per_100_words,
        p.imperative_rate,
        p.certainty_rate_per_100_words,
        p.hedge_rate_per_100_words,
        p.causal_connective_rate_per_100_words,
        p.contrast_connective_rate_per_100_words,
        deferred(p.formality_score, "a POS tagger"),
        deferred(p.sentiment_polarity, "an affect lexicon"),
        deferred(p.sentiment_subjectivity, "an affect lexicon"),
    )
}

fn run_profile(cmd: ProfileCmd) -> Result<(), String> {
    let text = read_input(&cmd.path)?;
    let profile = compute_linguistic_profile(&text);
    if cmd.json {
        let json = serde_json::to_string_pretty(&profile)
            .map_err(|e| format!("serializing profile: {e}"))?;
        println!("{json}");
    } else {
        println!(
            "{}",
            render_profile(&profile, larc_lexicon_engine::word_count(&text))
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Everything below needs storage + embeddings.
// ---------------------------------------------------------------------------

#[cfg(feature = "sqlite-backend")]
mod store {
    use super::*;

    use std::path::PathBuf;

    use chrono::{DateTime, Utc};
    use clap::ValueEnum;
    use larc_lexicon_engine::backend::SqliteEngine;
    use larc_lexicon_engine::{
        extract_human_turns, word_count, Confidence, LexiconEngine, PatternCategory, Register,
        SourceKind, SourceRef, VoicePattern, VoiceSample,
    };
    use uuid::Uuid;

    #[derive(Args)]
    pub struct StoreOpts {
        /// Lexicon database path. Defaults to $LARC_LEXICON_DB, else
        /// ~/.larc-lexicon/voice.db.
        #[arg(long)]
        pub db: Option<PathBuf>,
    }

    impl StoreOpts {
        pub fn open(&self) -> Result<SqliteEngine, String> {
            let path = self.resolve()?;
            SqliteEngine::open(&path).map_err(|e| format!("opening {}: {e}", path.display()))
        }

        fn resolve(&self) -> Result<PathBuf, String> {
            if let Some(p) = &self.db {
                return Ok(p.clone());
            }
            if let Some(p) = std::env::var_os("LARC_LEXICON_DB") {
                return Ok(PathBuf::from(p));
            }
            let home = std::env::var_os("HOME")
                .ok_or("HOME is unset — pass --db or set LARC_LEXICON_DB")?;
            let dir = PathBuf::from(home).join(".larc-lexicon");
            std::fs::create_dir_all(&dir)
                .map_err(|e| format!("creating {}: {e}", dir.display()))?;
            Ok(dir.join("voice.db"))
        }
    }

    // clap `ValueEnum` mirrors of the schema enums. Deliberately duplicated in
    // the binary so the library never grows a clap dependency for the sake of
    // argument parsing it does not perform.

    #[derive(Clone, Copy, ValueEnum)]
    pub enum ConfidenceArg {
        Verbatim,
        Relayed,
        Paraphrased,
    }

    impl From<ConfidenceArg> for Confidence {
        fn from(v: ConfidenceArg) -> Self {
            match v {
                ConfidenceArg::Verbatim => Confidence::Verbatim,
                ConfidenceArg::Relayed => Confidence::Relayed,
                ConfidenceArg::Paraphrased => Confidence::Paraphrased,
            }
        }
    }

    #[derive(Clone, Copy, ValueEnum)]
    pub enum SourceKindArg {
        ClaudeCodeSession,
        HelixTranscript,
        HelixDogfood,
        Manual,
    }

    impl From<SourceKindArg> for SourceKind {
        fn from(v: SourceKindArg) -> Self {
            match v {
                SourceKindArg::ClaudeCodeSession => SourceKind::ClaudeCodeSession,
                SourceKindArg::HelixTranscript => SourceKind::HelixTranscript,
                SourceKindArg::HelixDogfood => SourceKind::HelixDogfood,
                SourceKindArg::Manual => SourceKind::Manual,
            }
        }
    }

    #[derive(Clone, Copy, ValueEnum)]
    pub enum RegisterArg {
        Terse,
        Conversational,
        Technical,
        Directive,
        Reflective,
    }

    impl From<RegisterArg> for Register {
        fn from(v: RegisterArg) -> Self {
            match v {
                RegisterArg::Terse => Register::Terse,
                RegisterArg::Conversational => Register::Conversational,
                RegisterArg::Technical => Register::Technical,
                RegisterArg::Directive => Register::Directive,
                RegisterArg::Reflective => Register::Reflective,
            }
        }
    }

    #[derive(Clone, Copy, ValueEnum)]
    pub enum PatternCategoryArg {
        Opener,
        SentenceConstruction,
        ToneRule,
        Vocabulary,
        AntiPattern,
    }

    impl From<PatternCategoryArg> for PatternCategory {
        fn from(v: PatternCategoryArg) -> Self {
            match v {
                PatternCategoryArg::Opener => PatternCategory::Opener,
                PatternCategoryArg::SentenceConstruction => PatternCategory::SentenceConstruction,
                PatternCategoryArg::ToneRule => PatternCategory::ToneRule,
                PatternCategoryArg::Vocabulary => PatternCategory::Vocabulary,
                PatternCategoryArg::AntiPattern => PatternCategory::AntiPattern,
            }
        }
    }

    /// Stable id for a sample, derived from its origin and its exact text.
    ///
    /// Deterministic on purpose: ingest commands are meant to be re-run as
    /// sources grow, and the backend upserts on `id`. A random v4 id would
    /// make every re-run duplicate every previously-ingested turn; a v5 id
    /// makes re-ingest idempotent, and makes an edited turn a genuinely new
    /// sample rather than a silent overwrite.
    fn stable_id(scope: &str, locator: &str, text: &str) -> Uuid {
        let key = format!("larc/{scope}/{locator}/{text}");
        Uuid::new_v5(&Uuid::NAMESPACE_URL, key.as_bytes())
    }

    #[allow(clippy::too_many_arguments)]
    fn build_sample(
        id: Uuid,
        author: &str,
        text: String,
        source: SourceRef,
        captured_at: DateTime<Utc>,
        confidence: Confidence,
        register: Option<Register>,
        tags: Vec<String>,
    ) -> VoiceSample {
        let profile = compute_linguistic_profile(&text);
        let words = word_count(&text);
        VoiceSample {
            id,
            author: author.to_string(),
            text,
            source,
            captured_at,
            word_count: words,
            register,
            tags,
            confidence,
            profile,
        }
    }

    // -- ingest ------------------------------------------------------------

    #[derive(Args)]
    pub struct IngestCmd {
        #[command(subcommand)]
        source: IngestSource,
    }

    #[derive(Subcommand)]
    enum IngestSource {
        /// Ingest one file as a single sample.
        File(IngestFileCmd),
        /// Ingest human-authored turns from Claude Code session transcripts.
        ClaudeSessions(IngestSessionsCmd),
    }

    #[derive(Args)]
    struct IngestFileCmd {
        /// File whose entire contents become one sample.
        path: PathBuf,
        #[arg(long)]
        author: String,
        #[arg(long, value_enum, default_value = "verbatim")]
        confidence: ConfidenceArg,
        #[arg(long, value_enum, default_value = "manual")]
        source_kind: SourceKindArg,
        #[arg(long, value_enum)]
        register: Option<RegisterArg>,
        #[arg(long)]
        project: Option<String>,
        #[arg(long, value_delimiter = ',')]
        tags: Vec<String>,
        #[command(flatten)]
        store: StoreOpts,
    }

    #[derive(Args)]
    struct IngestSessionsCmd {
        /// A `.jsonl` transcript, or a directory of them
        /// (e.g. ~/.claude/projects/<slug>).
        path: PathBuf,
        /// Author to attribute the human turns to.
        #[arg(long)]
        author: String,
        #[arg(long)]
        project: Option<String>,
        /// Skip turns shorter than this. Very short turns ("ok", "yes, ship
        /// it") carry no usable stylometric signal but do skew corpus-level
        /// averages.
        #[arg(long, default_value_t = 5)]
        min_words: u32,
        /// Report what would be ingested without writing to the lexicon.
        #[arg(long)]
        dry_run: bool,
        #[command(flatten)]
        store: StoreOpts,
    }

    fn collect_jsonl(path: &PathBuf) -> Result<Vec<PathBuf>, String> {
        if path.is_file() {
            return Ok(vec![path.clone()]);
        }
        if !path.is_dir() {
            return Err(format!(
                "{} is neither a file nor a directory",
                path.display()
            ));
        }
        let mut out = Vec::new();
        for entry in
            std::fs::read_dir(path).map_err(|e| format!("reading {}: {e}", path.display()))?
        {
            let entry = entry.map_err(|e| format!("reading {}: {e}", path.display()))?;
            let p = entry.path();
            if p.extension().and_then(|e| e.to_str()) == Some("jsonl") {
                out.push(p);
            }
        }
        out.sort();
        if out.is_empty() {
            return Err(format!("no .jsonl transcripts found in {}", path.display()));
        }
        Ok(out)
    }

    pub async fn run_ingest(cmd: IngestCmd) -> Result<(), String> {
        match cmd.source {
            IngestSource::File(c) => run_ingest_file(c).await,
            IngestSource::ClaudeSessions(c) => run_ingest_sessions(c).await,
        }
    }

    async fn run_ingest_file(cmd: IngestFileCmd) -> Result<(), String> {
        let text = std::fs::read_to_string(&cmd.path)
            .map_err(|e| format!("reading {}: {e}", cmd.path.display()))?;
        let text = text.trim().to_string();
        if text.is_empty() {
            return Err(format!("{} is empty", cmd.path.display()));
        }

        let locator = cmd.path.display().to_string();
        let source = SourceRef {
            kind: cmd.source_kind.into(),
            project: cmd.project,
            session_id: None,
            locator: locator.clone(),
        };
        let sample = build_sample(
            stable_id("file", &locator, &text),
            &cmd.author,
            text,
            source,
            Utc::now(),
            cmd.confidence.into(),
            cmd.register.map(Into::into),
            cmd.tags,
        );

        let engine = cmd.store.open()?;
        let n = engine
            .ingest(&[sample])
            .await
            .map_err(|e| format!("ingest failed: {e}"))?;
        println!("ingested {n} sample(s)");
        Ok(())
    }

    async fn run_ingest_sessions(cmd: IngestSessionsCmd) -> Result<(), String> {
        let files = collect_jsonl(&cmd.path)?;

        let mut samples = Vec::new();
        let mut scanned_turns = 0usize;
        let mut skipped_short = 0usize;

        for file in &files {
            let raw = std::fs::read_to_string(file)
                .map_err(|e| format!("reading {}: {e}", file.display()))?;
            let locator = file.display().to_string();

            for turn in extract_human_turns(&raw) {
                scanned_turns += 1;
                if word_count(&turn.text) < cmd.min_words {
                    skipped_short += 1;
                    continue;
                }
                let captured_at = turn
                    .timestamp
                    .as_deref()
                    .and_then(|t| DateTime::parse_from_rfc3339(t).ok())
                    .map(|t| t.with_timezone(&Utc))
                    .unwrap_or_else(Utc::now);
                let scope_locator = turn.session_id.clone().unwrap_or_else(|| locator.clone());
                let source = SourceRef {
                    kind: SourceKind::ClaudeCodeSession,
                    project: cmd.project.clone(),
                    session_id: turn.session_id.clone(),
                    locator: locator.clone(),
                };
                samples.push(build_sample(
                    stable_id("claude-session", &scope_locator, &turn.text),
                    &cmd.author,
                    turn.text,
                    source,
                    captured_at,
                    // These are the author's own keystrokes, captured as typed.
                    Confidence::Verbatim,
                    None,
                    Vec::new(),
                ));
            }
        }

        println!(
            "{} transcript(s): {scanned_turns} human turn(s) found, {skipped_short} below \
             --min-words {}, {} to ingest",
            files.len(),
            cmd.min_words,
            samples.len()
        );

        if cmd.dry_run {
            for s in samples.iter().take(10) {
                println!("  [{} words] {}", s.word_count, first_line(&s.text, 96));
            }
            if samples.len() > 10 {
                println!("  ... and {} more", samples.len() - 10);
            }
            println!("dry run — nothing written");
            return Ok(());
        }
        if samples.is_empty() {
            return Ok(());
        }

        let engine = cmd.store.open()?;
        let n = engine
            .ingest(&samples)
            .await
            .map_err(|e| format!("ingest failed: {e}"))?;
        println!("ingested {n} sample(s)");
        Ok(())
    }

    fn first_line(text: &str, max: usize) -> String {
        let line = text.lines().next().unwrap_or_default().trim();
        let truncated: String = line.chars().take(max).collect();
        if line.chars().count() > max {
            format!("{truncated}…")
        } else {
            truncated
        }
    }

    // -- search ------------------------------------------------------------

    #[derive(Args)]
    pub struct SearchCmd {
        /// Free-text query, matched by embedding similarity.
        query: String,
        #[arg(long)]
        author: Option<String>,
        #[arg(long, default_value_t = 5)]
        top_k: usize,
        #[arg(long)]
        json: bool,
        #[command(flatten)]
        store: StoreOpts,
    }

    pub async fn run_search(cmd: SearchCmd) -> Result<(), String> {
        let engine = cmd.store.open()?;
        let hits = engine
            .search(&cmd.query, cmd.author.as_deref(), cmd.top_k)
            .await
            .map_err(|e| format!("search failed: {e}"))?;

        if cmd.json {
            let json =
                serde_json::to_string_pretty(&hits).map_err(|e| format!("serializing: {e}"))?;
            println!("{json}");
            return Ok(());
        }

        if hits.is_empty() {
            println!("no matches");
            return Ok(());
        }
        for (i, s) in hits.iter().enumerate() {
            println!(
                "{}. {} · {:?} · {} words · {}",
                i + 1,
                s.author,
                s.confidence,
                s.word_count,
                s.source.locator
            );
            println!("   {}", first_line(&s.text, 120));
        }
        Ok(())
    }

    // -- patterns ----------------------------------------------------------

    #[derive(Args)]
    pub struct PatternsCmd {
        #[command(subcommand)]
        action: PatternsAction,
    }

    #[derive(Subcommand)]
    enum PatternsAction {
        /// List an author's distilled patterns.
        List(PatternsListCmd),
        /// Record a distilled pattern.
        Add(PatternsAddCmd),
    }

    #[derive(Args)]
    struct PatternsListCmd {
        author: String,
        #[arg(long)]
        json: bool,
        #[command(flatten)]
        store: StoreOpts,
    }

    #[derive(Args)]
    struct PatternsAddCmd {
        #[arg(long)]
        author: String,
        #[arg(long, value_enum)]
        category: PatternCategoryArg,
        /// The rule itself, in plain language.
        #[arg(long)]
        description: String,
        /// Id of a sample this was drawn from. Repeatable.
        #[arg(long = "example-id")]
        example_ids: Vec<Uuid>,
        /// Record as a habit to recognize but never reproduce (e.g. a recurring
        /// typo), rather than as authentic voice worth replicating.
        #[arg(long)]
        anti_pattern: bool,
        #[command(flatten)]
        store: StoreOpts,
    }

    pub async fn run_patterns(cmd: PatternsCmd) -> Result<(), String> {
        match cmd.action {
            PatternsAction::List(c) => {
                let engine = c.store.open()?;
                let patterns = engine
                    .patterns_for(&c.author)
                    .await
                    .map_err(|e| format!("query failed: {e}"))?;
                if c.json {
                    let json = serde_json::to_string_pretty(&patterns)
                        .map_err(|e| format!("serializing: {e}"))?;
                    println!("{json}");
                } else if patterns.is_empty() {
                    println!("no patterns recorded for {}", c.author);
                } else {
                    for p in &patterns {
                        let marker = if p.replicate { "replicate" } else { "avoid" };
                        println!("{:?} [{marker}] {}", p.category, p.description);
                        println!("  id {} · {} example(s)", p.id, p.example_ids.len());
                    }
                }
                Ok(())
            }
            PatternsAction::Add(c) => {
                let pattern = VoicePattern {
                    id: stable_id("pattern", &c.author, &c.description),
                    author: c.author,
                    category: c.category.into(),
                    description: c.description,
                    example_ids: c.example_ids,
                    replicate: !c.anti_pattern,
                };
                let engine = c.store.open()?;
                let n = engine
                    .save_patterns(&[pattern])
                    .await
                    .map_err(|e| format!("save failed: {e}"))?;
                println!("saved {n} pattern(s)");
                Ok(())
            }
        }
    }
}

#[cfg(feature = "sqlite-backend")]
use store::{IngestCmd, PatternsCmd, SearchCmd};

// ---------------------------------------------------------------------------

async fn run(cli: Cli) -> Result<(), String> {
    match cli.command {
        Command::Profile(cmd) => run_profile(cmd),
        #[cfg(feature = "sqlite-backend")]
        Command::Ingest(cmd) => store::run_ingest(cmd).await,
        #[cfg(feature = "sqlite-backend")]
        Command::Search(cmd) => store::run_search(cmd).await,
        #[cfg(feature = "sqlite-backend")]
        Command::Patterns(cmd) => store::run_patterns(cmd).await,
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    match run(Cli::parse()).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("larc: {e}");
            ExitCode::FAILURE
        }
    }
}
