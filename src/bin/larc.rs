//! `larc` — command-line interface to the lexicon engine.
//!
//! Subcommands split by what they actually need. `profile` requires nothing
//! but the core crate, so it is always present. Everything that reads or
//! writes a lexicon needs storage and embeddings and is therefore compiled
//! only with the `sqlite-backend` feature — rather than shipping commands that
//! exist in `--help` but fail at runtime.

use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};
#[cfg(feature = "sqlite-backend")]
use larc_lexicon_engine::CorpusProfile;
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
    /// Correctly pooled corpus statistics for one author's stored samples.
    #[cfg(feature = "sqlite-backend")]
    Stats(StatsCmd),
    /// Derive candidate voice patterns from an author's stored samples.
    #[cfg(feature = "sqlite-backend")]
    Distill(DistillCmd),
    /// Render a ready-to-paste voice brief an LLM can condition on.
    #[cfg(feature = "sqlite-backend")]
    StyleGuide(StyleGuideCmd),
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

/// The measurement fields `LinguisticProfile` and `CorpusProfile` share —
/// everything except `CorpusProfile`'s corpus-only `document_count` /
/// `total_words`. `larc profile` and `larc stats` both render through this
/// one struct so a future field addition can't update one display and miss
/// the other, the way the two format strings here had already drifted.
struct DisplayProfile {
    type_token_ratio: f32,
    avg_word_length: f32,
    hapax_legomena_ratio: f32,
    avg_sentence_length: f32,
    sentence_length_stddev: f32,
    comma_rate_per_100_words: f32,
    semicolon_rate_per_100_words: f32,
    dash_rate_per_100_words: f32,
    question_rate_per_100_words: f32,
    exclamation_rate_per_100_words: f32,
    flesch_reading_ease: f32,
    flesch_kincaid_grade: f32,
    gunning_fog_index: f32,
    first_person_singular_rate_per_100_words: f32,
    first_person_plural_rate_per_100_words: f32,
    second_person_rate_per_100_words: f32,
    imperative_rate: f32,
    certainty_rate_per_100_words: f32,
    hedge_rate_per_100_words: f32,
    causal_connective_rate_per_100_words: f32,
    contrast_connective_rate_per_100_words: f32,
    formality_score: Option<f32>,
    sentiment_polarity: Option<f32>,
    sentiment_subjectivity: Option<f32>,
}

impl From<&LinguisticProfile> for DisplayProfile {
    fn from(p: &LinguisticProfile) -> Self {
        Self {
            type_token_ratio: p.type_token_ratio,
            avg_word_length: p.avg_word_length,
            hapax_legomena_ratio: p.hapax_legomena_ratio,
            avg_sentence_length: p.avg_sentence_length,
            sentence_length_stddev: p.sentence_length_stddev,
            comma_rate_per_100_words: p.comma_rate_per_100_words,
            semicolon_rate_per_100_words: p.semicolon_rate_per_100_words,
            dash_rate_per_100_words: p.dash_rate_per_100_words,
            question_rate_per_100_words: p.question_rate_per_100_words,
            exclamation_rate_per_100_words: p.exclamation_rate_per_100_words,
            flesch_reading_ease: p.flesch_reading_ease,
            flesch_kincaid_grade: p.flesch_kincaid_grade,
            gunning_fog_index: p.gunning_fog_index,
            first_person_singular_rate_per_100_words: p.first_person_singular_rate_per_100_words,
            first_person_plural_rate_per_100_words: p.first_person_plural_rate_per_100_words,
            second_person_rate_per_100_words: p.second_person_rate_per_100_words,
            imperative_rate: p.imperative_rate,
            certainty_rate_per_100_words: p.certainty_rate_per_100_words,
            hedge_rate_per_100_words: p.hedge_rate_per_100_words,
            causal_connective_rate_per_100_words: p.causal_connective_rate_per_100_words,
            contrast_connective_rate_per_100_words: p.contrast_connective_rate_per_100_words,
            formality_score: p.formality_score,
            sentiment_polarity: p.sentiment_polarity,
            sentiment_subjectivity: p.sentiment_subjectivity,
        }
    }
}

#[cfg(feature = "sqlite-backend")]
impl From<&CorpusProfile> for DisplayProfile {
    fn from(p: &CorpusProfile) -> Self {
        Self {
            type_token_ratio: p.type_token_ratio,
            avg_word_length: p.avg_word_length,
            hapax_legomena_ratio: p.hapax_legomena_ratio,
            avg_sentence_length: p.avg_sentence_length,
            sentence_length_stddev: p.sentence_length_stddev,
            comma_rate_per_100_words: p.comma_rate_per_100_words,
            semicolon_rate_per_100_words: p.semicolon_rate_per_100_words,
            dash_rate_per_100_words: p.dash_rate_per_100_words,
            question_rate_per_100_words: p.question_rate_per_100_words,
            exclamation_rate_per_100_words: p.exclamation_rate_per_100_words,
            flesch_reading_ease: p.flesch_reading_ease,
            flesch_kincaid_grade: p.flesch_kincaid_grade,
            gunning_fog_index: p.gunning_fog_index,
            first_person_singular_rate_per_100_words: p.first_person_singular_rate_per_100_words,
            first_person_plural_rate_per_100_words: p.first_person_plural_rate_per_100_words,
            second_person_rate_per_100_words: p.second_person_rate_per_100_words,
            imperative_rate: p.imperative_rate,
            certainty_rate_per_100_words: p.certainty_rate_per_100_words,
            hedge_rate_per_100_words: p.hedge_rate_per_100_words,
            causal_connective_rate_per_100_words: p.causal_connective_rate_per_100_words,
            contrast_connective_rate_per_100_words: p.contrast_connective_rate_per_100_words,
            formality_score: p.formality_score,
            sentiment_polarity: p.sentiment_polarity,
            sentiment_subjectivity: p.sentiment_subjectivity,
        }
    }
}

fn render_display_profile(header: &str, p: &DisplayProfile) -> String {
    format!(
        "{header}

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

fn render_profile(p: &LinguisticProfile, words: u32) -> String {
    render_display_profile(&format!("Words: {words}"), &p.into())
}

/// `larc stats`'s renderer — same body as `render_profile`, via
/// [`render_display_profile`], with the corpus-only header fields
/// `CorpusProfile` carries that a single `LinguisticProfile` has no use for.
#[cfg(feature = "sqlite-backend")]
fn render_corpus_profile(p: &CorpusProfile) -> String {
    render_display_profile(
        &format!("Documents: {}   Words: {}", p.document_count, p.total_words),
        &p.into(),
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
        aggregate_corpus_profile, extract_human_turns, flag_pasted_content, word_count, Confidence,
        LexiconEngine, PatternCategory, Register, SourceKind, SourceRef, VoicePattern, VoiceSample,
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
        larc_lexicon_engine::stable_uuid(&[scope, locator, text])
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
        /// One or more files, each ingested as a single sample sharing the
        /// same --author/--confidence/--source-kind/--tags. A single
        /// `larc ingest file a.txt b.txt c.txt` opens the lexicon once and
        /// embeds every text in one batch, instead of paying per-process
        /// startup and per-call embedding overhead for each file.
        #[arg(required = true)]
        paths: Vec<PathBuf>,
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
        /// Drop turns flagged as likely pasted material (see the warning
        /// list printed by default) instead of ingesting them anyway.
        #[arg(long)]
        exclude_pasted: bool,
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
        let mut samples = Vec::with_capacity(cmd.paths.len());
        for path in &cmd.paths {
            let text = std::fs::read_to_string(path)
                .map_err(|e| format!("reading {}: {e}", path.display()))?;
            let text = text.trim().to_string();
            if text.is_empty() {
                return Err(format!("{} is empty", path.display()));
            }

            let locator = path.display().to_string();
            let source = SourceRef {
                kind: cmd.source_kind.into(),
                project: cmd.project.clone(),
                session_id: None,
                locator: locator.clone(),
            };
            samples.push(build_sample(
                stable_id("file", &locator, &text),
                &cmd.author,
                text,
                source,
                Utc::now(),
                cmd.confidence.into(),
                cmd.register.map(Into::into),
                cmd.tags.clone(),
            ));
        }

        let engine = cmd.store.open()?;
        let n = engine
            .ingest(&samples)
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

            for (turn_index, turn) in extract_human_turns(&raw).into_iter().enumerate() {
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
                // `turn_index` disambiguates two turns with identical text in
                // the same session (e.g. two separate "yes" confirmations) —
                // without it they'd hash to the same id and silently
                // overwrite each other. Stable across re-runs of the same
                // file because `extract_human_turns` always returns turns in
                // the order they appear in the (append-only) transcript.
                let scope_locator = format!(
                    "{}#{turn_index}",
                    turn.session_id.clone().unwrap_or_else(|| locator.clone())
                );
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

        // Paste detection is a batch property (length-outlier is relative to
        // the rest of this ingest), so it runs once here rather than per turn
        // during collection above.
        let texts: Vec<&str> = samples.iter().map(|s| s.text.as_str()).collect();
        let flags = flag_pasted_content(&texts);
        let flagged: Vec<usize> = flags
            .iter()
            .enumerate()
            .filter(|(_, f)| f.any())
            .map(|(i, _)| i)
            .collect();

        println!(
            "{} transcript(s): {scanned_turns} human turn(s) found, {skipped_short} below \
             --min-words {}, {} to ingest",
            files.len(),
            cmd.min_words,
            samples.len()
        );

        if !flagged.is_empty() {
            let verb = if cmd.exclude_pasted {
                "excluding"
            } else {
                "ingesting anyway — rerun with --exclude-pasted to drop these"
            };
            println!(
                "{} sample(s) look like pasted material ({verb}):",
                flagged.len()
            );
            for &i in &flagged {
                // `(false, false)` can't occur today — `flagged` is
                // pre-filtered by `PasteSignal::any()` — but a display
                // label, not a panic, is the safe choice if that filter
                // is ever refactored to include unflagged indices.
                let reason = match (flags[i].length_outlier, flags[i].structural_markup) {
                    (true, true) => "length+structure",
                    (true, false) => "length",
                    (false, true) => "structure",
                    (false, false) => "flagged",
                };
                println!(
                    "  [{reason}, {}w] {}",
                    samples[i].word_count,
                    first_line(&samples[i].text, 80)
                );
            }
        }

        if cmd.exclude_pasted && !flagged.is_empty() {
            let excluded: std::collections::HashSet<usize> = flagged.into_iter().collect();
            let mut i = 0usize;
            samples.retain(|_| {
                let keep = !excluded.contains(&i);
                i += 1;
                keep
            });
        }

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

    /// Truncate `s` to `max` chars, appending an ellipsis if anything was
    /// cut. Shared by every preview/quote renderer so a future fix to the
    /// truncation itself (a Unicode-boundary case, an off-by-one) can't be
    /// applied to one caller and missed on another.
    fn truncate_with_ellipsis(s: &str, max: usize) -> String {
        let truncated: String = s.chars().take(max).collect();
        if s.chars().count() > max {
            format!("{truncated}…")
        } else {
            truncated
        }
    }

    fn first_line(text: &str, max: usize) -> String {
        let line = text.lines().next().unwrap_or_default().trim();
        truncate_with_ellipsis(line, max)
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
        /// Delete a pattern by id, or every pattern for an author.
        Delete(PatternsDeleteCmd),
        /// Revise an existing pattern's wording, category, replicate flag,
        /// or evidence, keeping its id.
        Update(PatternsUpdateCmd),
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

    #[derive(Args)]
    struct PatternsDeleteCmd {
        /// Pattern id to delete. Omit when using --author with --all.
        id: Option<Uuid>,
        /// Author whose patterns to delete — only meaningful with --all.
        #[arg(long, requires = "all")]
        author: Option<String>,
        /// Delete every pattern for --author, instead of one by id.
        #[arg(long, requires = "author")]
        all: bool,
        #[command(flatten)]
        store: StoreOpts,
    }

    /// Revise an existing pattern in place.
    ///
    /// Deliberately identity-addressed rather than content-addressed: `add`
    /// derives a pattern's id from `(author, category, description)`, so
    /// changing the wording there creates a *new* pattern, orphaning the
    /// old one. `update` takes the id as input and keeps it fixed no matter
    /// what else changes — the escape hatch for fixing a typo or tightening
    /// a description without losing the pattern's identity, its existing
    /// `example_ids`, or having to delete-then-re-add by hand.
    #[derive(Args)]
    struct PatternsUpdateCmd {
        /// Id of the pattern to update.
        id: Uuid,
        /// Author the pattern belongs to.
        #[arg(long)]
        author: String,
        /// New description. Omit to keep the existing one.
        #[arg(long)]
        description: Option<String>,
        /// New category. Omit to keep the existing one.
        #[arg(long, value_enum)]
        category: Option<PatternCategoryArg>,
        /// Mark as an anti-pattern (never reproduce).
        #[arg(long, conflicts_with = "replicate")]
        anti_pattern: bool,
        /// Mark as a pattern to replicate.
        #[arg(long)]
        replicate: bool,
        /// Additional sample id to link as evidence. Repeatable, and added
        /// to the existing example_ids rather than replacing them.
        #[arg(long = "add-example-id")]
        add_example_ids: Vec<Uuid>,
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
                let category: PatternCategory = c.category.into();
                // Category is folded into the id, not just author+description:
                // two different patterns for the same author can legitimately
                // share description wording (plausible when iterating on how
                // to phrase one) while differing in category or replicate
                // status — without this they'd hash to the same id and one
                // would silently overwrite the other via INSERT OR REPLACE.
                let pattern = VoicePattern {
                    id: stable_id(
                        "pattern",
                        &c.author,
                        &format!("{category:?}/{}", c.description),
                    ),
                    author: c.author,
                    category,
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
            PatternsAction::Delete(c) => {
                let engine = c.store.open()?;
                let ids: Vec<Uuid> = if c.all {
                    // `requires = "author"` on --all guarantees this is Some.
                    let author = c.author.as_deref().unwrap_or_default();
                    engine
                        .patterns_for(author)
                        .await
                        .map_err(|e| format!("query failed: {e}"))?
                        .into_iter()
                        .map(|p| p.id)
                        .collect()
                } else if let Some(id) = c.id {
                    vec![id]
                } else {
                    return Err(
                        "specify a pattern id, or --author X --all to delete every pattern \
                         for that author"
                            .to_string(),
                    );
                };
                let n = engine
                    .delete_patterns(&ids)
                    .await
                    .map_err(|e| format!("delete failed: {e}"))?;
                println!("deleted {n} pattern(s)");
                Ok(())
            }
            PatternsAction::Update(c) => {
                let engine = c.store.open()?;
                let existing = engine
                    .patterns_for(&c.author)
                    .await
                    .map_err(|e| format!("query failed: {e}"))?
                    .into_iter()
                    .find(|p| p.id == c.id)
                    .ok_or_else(|| {
                        format!("no pattern with id {} for author {}", c.id, c.author)
                    })?;

                let mut example_ids = existing.example_ids;
                example_ids.extend(c.add_example_ids);

                let updated = VoicePattern {
                    id: c.id,
                    author: c.author,
                    category: c.category.map(Into::into).unwrap_or(existing.category),
                    description: c.description.unwrap_or(existing.description),
                    example_ids,
                    replicate: if c.anti_pattern {
                        false
                    } else if c.replicate {
                        true
                    } else {
                        existing.replicate
                    },
                };
                let n = engine
                    .save_patterns(&[updated])
                    .await
                    .map_err(|e| format!("save failed: {e}"))?;
                println!("updated {n} pattern(s)");
                Ok(())
            }
        }
    }

    // -- stats ---------------------------------------------------------

    #[derive(Args)]
    pub struct StatsCmd {
        /// Author to compute statistics for. Omit for the whole lexicon.
        author: Option<String>,
        /// Only include samples carrying this tag (e.g. "professional") —
        /// compares registers without a hand-rolled script.
        #[arg(long)]
        tag: Option<String>,
        #[arg(long)]
        json: bool,
        #[command(flatten)]
        store: StoreOpts,
    }

    pub async fn run_stats(cmd: StatsCmd) -> Result<(), String> {
        let engine = cmd.store.open()?;
        let samples = engine
            .samples_for(cmd.author.as_deref())
            .await
            .map_err(|e| format!("query failed: {e}"))?;

        let filtered: Vec<&VoiceSample> = samples
            .iter()
            .filter(|s| match &cmd.tag {
                Some(t) => s.tags.iter().any(|tag| tag == t),
                None => true,
            })
            .collect();
        let texts: Vec<&str> = filtered.iter().map(|s| s.text.as_str()).collect();
        let profile = aggregate_corpus_profile(texts);

        if cmd.json {
            let json =
                serde_json::to_string_pretty(&profile).map_err(|e| format!("serializing: {e}"))?;
            println!("{json}");
        } else if profile.document_count == 0 {
            println!(
                "no samples found{}",
                cmd.author
                    .as_deref()
                    .map(|a| format!(" for {a}"))
                    .unwrap_or_default()
            );
        } else {
            println!("{}", render_corpus_profile(&profile));
        }
        Ok(())
    }

    // -- distill ---------------------------------------------------------

    #[derive(Args)]
    pub struct DistillCmd {
        /// Author to derive patterns for.
        author: String,
        /// Only include samples carrying this tag (e.g. "professional") —
        /// derive patterns for one register without a hand-rolled script.
        #[arg(long)]
        tag: Option<String>,
        /// Report what would be derived without writing to the lexicon.
        #[arg(long)]
        dry_run: bool,
        #[command(flatten)]
        store: StoreOpts,
    }

    pub async fn run_distill(cmd: DistillCmd) -> Result<(), String> {
        let engine = cmd.store.open()?;
        let samples = engine
            .samples_for(Some(&cmd.author))
            .await
            .map_err(|e| format!("query failed: {e}"))?;

        let filtered: Vec<VoiceSample> = samples
            .into_iter()
            .filter(|s| match &cmd.tag {
                Some(t) => s.tags.iter().any(|tag| tag == t),
                None => true,
            })
            .collect();
        let sample_count = filtered.len();

        let patterns = larc_lexicon_engine::distill_patterns(&cmd.author, &filtered);

        if patterns.is_empty() {
            println!(
                "no patterns derived from {sample_count} sample(s) — corpus is below the \
                 minimum size, or no signal cleared its threshold"
            );
            return Ok(());
        }

        for p in &patterns {
            let marker = if p.replicate { "replicate" } else { "avoid" };
            println!("{:?} [{marker}] {}", p.category, p.description);
            println!("  {} example(s)", p.example_ids.len());
        }

        if cmd.dry_run {
            println!("dry run — nothing written");
            return Ok(());
        }

        let n = engine
            .save_patterns(&patterns)
            .await
            .map_err(|e| format!("save failed: {e}"))?;
        println!("saved {n} pattern(s)");
        Ok(())
    }

    // -- style-guide -----------------------------------------------------

    #[derive(Args)]
    pub struct StyleGuideCmd {
        /// Author to build a style guide for.
        author: String,
        #[command(flatten)]
        store: StoreOpts,
    }

    const MAX_QUOTES_PER_PATTERN: usize = 3;
    const MAX_QUOTE_CHARS: usize = 180;

    /// Human-readable heading for a `PatternCategory` — the enum's `Debug`
    /// output (`SentenceConstruction`, `AntiPattern`) reads fine in `larc
    /// patterns list` but not as a markdown section header.
    fn category_heading(c: PatternCategory) -> &'static str {
        match c {
            PatternCategory::Opener => "Opener",
            PatternCategory::SentenceConstruction => "Sentence Construction",
            PatternCategory::ToneRule => "Tone",
            PatternCategory::Vocabulary => "Vocabulary",
            PatternCategory::AntiPattern => "Anti-Pattern",
        }
    }

    /// Collapse a sample's text to one line and cap its length, so a long
    /// pasted turn doesn't dominate the brief the way it dominated the
    /// corpus's word count.
    fn quote(text: &str, max: usize) -> String {
        let collapsed: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
        truncate_with_ellipsis(&collapsed, max)
    }

    fn render_style_guide(
        author: &str,
        patterns: &[VoicePattern],
        samples_by_id: &std::collections::HashMap<Uuid, &VoiceSample>,
        document_count: usize,
        total_words: u32,
    ) -> String {
        let mut out = format!(
            "# Writing Voice: {author}\n\nDerived from {document_count} measured samples \
             ({total_words} words). Follow the Replicate patterns; recognize but never \
             reproduce anything under Avoid.\n"
        );

        for (heading, replicate) in [("Replicate", true), ("Avoid", false)] {
            let section: Vec<&VoicePattern> = patterns
                .iter()
                .filter(|p| p.replicate == replicate)
                .collect();
            if section.is_empty() {
                continue;
            }
            out.push_str(&format!("\n## {heading}\n"));
            for p in section {
                out.push_str(&format!(
                    "\n### {}\n{}\n",
                    category_heading(p.category),
                    p.description
                ));
                let quotes: Vec<&str> = p
                    .example_ids
                    .iter()
                    .filter_map(|id| samples_by_id.get(id))
                    .map(|s| s.text.as_str())
                    .take(MAX_QUOTES_PER_PATTERN)
                    .collect();
                for q in quotes {
                    out.push_str(&format!("- \"{}\"\n", quote(q, MAX_QUOTE_CHARS)));
                }
            }
        }
        out
    }

    pub async fn run_style_guide(cmd: StyleGuideCmd) -> Result<(), String> {
        let engine = cmd.store.open()?;
        let patterns = engine
            .patterns_for(&cmd.author)
            .await
            .map_err(|e| format!("query failed: {e}"))?;
        if patterns.is_empty() {
            println!(
                "no patterns recorded for {} — run `larc distill {}` or `larc patterns add` \
                 first",
                cmd.author, cmd.author
            );
            return Ok(());
        }

        let samples = engine
            .samples_for(Some(&cmd.author))
            .await
            .map_err(|e| format!("query failed: {e}"))?;
        let total_words: u32 = samples.iter().map(|s| s.word_count).sum();
        let samples_by_id: std::collections::HashMap<Uuid, &VoiceSample> =
            samples.iter().map(|s| (s.id, s)).collect();

        println!(
            "{}",
            render_style_guide(
                &cmd.author,
                &patterns,
                &samples_by_id,
                samples.len(),
                total_words
            )
        );
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Two turns with identical text in the same session (e.g. two
        /// separate "yes" confirmations) must not collide: without the
        /// per-turn index folded into `scope_locator`, both would hash to
        /// the same id and the second would silently overwrite the first
        /// via the backend's upsert.
        #[test]
        fn duplicate_text_at_different_turn_indices_gets_distinct_ids() {
            let session = "session-1";
            let text = "yes";
            let first = stable_id("claude-session", &format!("{session}#0"), text);
            let second = stable_id("claude-session", &format!("{session}#1"), text);
            assert_ne!(
                first, second,
                "identical text at different turn indices must not collide"
            );
        }

        /// The fix must not break idempotent re-ingest: the same turn at the
        /// same index in the same session, ingested twice (a re-run over an
        /// unchanged, append-only transcript), must still produce the same id.
        #[test]
        fn same_turn_reingested_at_the_same_index_is_still_idempotent() {
            let scope_locator = "session-1#3";
            let text = "Ship the CLI change now please.";
            let first = stable_id("claude-session", scope_locator, text);
            let second = stable_id("claude-session", scope_locator, text);
            assert_eq!(
                first, second,
                "re-ingesting the same turn must upsert, not duplicate"
            );
        }

        /// Two different patterns for the same author can legitimately share
        /// description wording while differing in category — folding
        /// category into the id key keeps them from colliding.
        #[test]
        fn patterns_with_shared_description_but_different_category_get_distinct_ids() {
            let author = "kevin";
            let description = "Never opens with an apology";
            let tone_rule_id = stable_id(
                "pattern",
                author,
                &format!("{:?}/{description}", PatternCategory::ToneRule),
            );
            let anti_pattern_id = stable_id(
                "pattern",
                author,
                &format!("{:?}/{description}", PatternCategory::AntiPattern),
            );
            assert_ne!(
                tone_rule_id, anti_pattern_id,
                "same description in a different category must not collide"
            );
        }

        fn test_sample(text: &str) -> VoiceSample {
            let profile = compute_linguistic_profile(text);
            VoiceSample {
                id: Uuid::new_v4(),
                author: "kevin".to_string(),
                text: text.to_string(),
                source: SourceRef {
                    kind: SourceKind::Manual,
                    project: None,
                    session_id: None,
                    locator: "test".to_string(),
                },
                captured_at: Utc::now(),
                word_count: word_count(text),
                register: None,
                tags: vec![],
                confidence: Confidence::Verbatim,
                profile,
            }
        }

        fn test_pattern(
            category: PatternCategory,
            description: &str,
            example_ids: Vec<Uuid>,
            replicate: bool,
        ) -> VoicePattern {
            VoicePattern {
                id: Uuid::new_v4(),
                author: "kevin".to_string(),
                category,
                description: description.to_string(),
                example_ids,
                replicate,
            }
        }

        #[test]
        fn category_heading_covers_every_variant_with_a_readable_label() {
            assert_eq!(category_heading(PatternCategory::Opener), "Opener");
            assert_eq!(
                category_heading(PatternCategory::SentenceConstruction),
                "Sentence Construction"
            );
            assert_eq!(category_heading(PatternCategory::ToneRule), "Tone");
            assert_eq!(category_heading(PatternCategory::Vocabulary), "Vocabulary");
            assert_eq!(
                category_heading(PatternCategory::AntiPattern),
                "Anti-Pattern"
            );
        }

        #[test]
        fn quote_collapses_whitespace_and_truncates_long_text() {
            assert_eq!(quote("one\ntwo   three", 100), "one two three");
            let long = "word ".repeat(50);
            let truncated = quote(&long, 10);
            assert_eq!(truncated.chars().count(), 11); // 10 chars + the ellipsis
            assert!(truncated.ends_with('…'));
        }

        #[test]
        fn quote_does_not_truncate_text_at_or_under_the_limit() {
            assert_eq!(quote("short", 100), "short");
        }

        #[test]
        fn render_style_guide_groups_by_replicate_not_by_category() {
            let s1 = test_sample("ok ship it");
            let replicate_pattern =
                test_pattern(PatternCategory::Opener, "Opens on ok", vec![s1.id], true);
            // An AntiPattern-categoried pattern that is still `replicate: true`
            // must land in Replicate, not Avoid — grouping is by the
            // `replicate` field, not by category naming.
            let odd_pattern = test_pattern(
                PatternCategory::AntiPattern,
                "Not actually an anti-pattern",
                vec![],
                true,
            );
            let avoid_pattern =
                test_pattern(PatternCategory::AntiPattern, "Never do this", vec![], false);

            let by_id: std::collections::HashMap<Uuid, &VoiceSample> =
                [(s1.id, &s1)].into_iter().collect();
            let out = render_style_guide(
                "kevin",
                &[replicate_pattern, odd_pattern, avoid_pattern],
                &by_id,
                1,
                s1.word_count,
            );

            let replicate_idx = out.find("## Replicate").unwrap();
            let avoid_idx = out.find("## Avoid").unwrap();
            assert!(replicate_idx < avoid_idx);
            assert!(out[replicate_idx..avoid_idx].contains("Not actually an anti-pattern"));
            assert!(out[avoid_idx..].contains("Never do this"));
            assert!(!out[avoid_idx..].contains("Not actually an anti-pattern"));
        }

        #[test]
        fn render_style_guide_resolves_example_ids_to_verbatim_quotes() {
            let s1 = test_sample("ok ship the release");
            let pattern = test_pattern(PatternCategory::Opener, "Opens on ok", vec![s1.id], true);
            let by_id: std::collections::HashMap<Uuid, &VoiceSample> =
                [(s1.id, &s1)].into_iter().collect();
            let out = render_style_guide("kevin", &[pattern], &by_id, 1, s1.word_count);
            assert!(out.contains("ok ship the release"));
        }

        #[test]
        fn render_style_guide_skips_dangling_example_ids_without_panicking() {
            let dangling_id = Uuid::new_v4();
            let pattern = test_pattern(
                PatternCategory::Opener,
                "Opens on ok",
                vec![dangling_id],
                true,
            );
            let by_id: std::collections::HashMap<Uuid, &VoiceSample> =
                std::collections::HashMap::new();
            // Must not panic despite the id resolving to nothing.
            let out = render_style_guide("kevin", &[pattern], &by_id, 0, 0);
            assert!(out.contains("Opens on ok"));
        }

        #[test]
        fn render_style_guide_caps_quotes_per_pattern() {
            let samples: Vec<VoiceSample> = (0..10)
                .map(|i| test_sample(&format!("ok ship number {i}")))
                .collect();
            let ids: Vec<Uuid> = samples.iter().map(|s| s.id).collect();
            let pattern = test_pattern(PatternCategory::Opener, "Opens on ok", ids, true);
            let by_id: std::collections::HashMap<Uuid, &VoiceSample> =
                samples.iter().map(|s| (s.id, s)).collect();
            let total_words: u32 = samples.iter().map(|s| s.word_count).sum();
            let out = render_style_guide("kevin", &[pattern], &by_id, samples.len(), total_words);
            let quote_lines = out.lines().filter(|l| l.starts_with("- \"")).count();
            assert_eq!(quote_lines, MAX_QUOTES_PER_PATTERN);
        }
    }
}

#[cfg(feature = "sqlite-backend")]
use store::{DistillCmd, IngestCmd, PatternsCmd, SearchCmd, StatsCmd, StyleGuideCmd};

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
        #[cfg(feature = "sqlite-backend")]
        Command::Stats(cmd) => store::run_stats(cmd).await,
        #[cfg(feature = "sqlite-backend")]
        Command::Distill(cmd) => store::run_distill(cmd).await,
        #[cfg(feature = "sqlite-backend")]
        Command::StyleGuide(cmd) => store::run_style_guide(cmd).await,
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
