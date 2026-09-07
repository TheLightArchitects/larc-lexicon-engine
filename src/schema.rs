use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// How sure we are that `VoiceSample::text` is genuinely, directly authored —
/// not summarized, not co-drafted, not relayed through an intermediary.
/// Trust tiering matters because a lexicon trained on paraphrased text
/// learns the paraphraser's voice, not the author's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Confidence {
    /// Directly typed/spoken by the author, captured verbatim.
    Verbatim,
    /// Dispatched under the author's name but possibly co-drafted or edited
    /// (e.g. a message relayed through another agent/system on their behalf).
    Relayed,
    /// A summary or paraphrase of what the author said — lowest trust, useful
    /// for topical search but never for style synthesis.
    Paraphrased,
}

/// Coarse register classification — useful as a retrieval filter (e.g. "show me
/// Directive-register samples") since the same author's voice shifts materially
/// by register, and blending registers when synthesizing a style guide produces
/// a muddled, inconsistent voice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Register {
    Terse,
    Conversational,
    Technical,
    Directive,
    Reflective,
}

/// Where a sample came from, kept precise enough to trace back to the exact
/// origin file/session for spot-checking or re-extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceRef {
    pub kind: SourceKind,
    pub project: Option<String>,
    pub session_id: Option<String>,
    /// Exact file path, URL, or other locator identifying the precise origin.
    pub locator: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceKind {
    ClaudeCodeSession,
    HelixTranscript,
    HelixDogfood,
    Manual,
}

/// Computed linguistic/stylometric features for one `VoiceSample`.
///
/// Every non-`Option` field is computed from tokenization and word-list
/// lookups alone — no POS tagging or trained model required, so the core
/// crate stays dependency-light and every number here is reproducible from
/// the text itself. `Option` fields name real, established metrics that
/// need more than tokenization (a POS tagger, an affect lexicon) and are
/// left `None` until a backend for them exists, rather than approximated
/// and presented as measured.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct LinguisticProfile {
    // --- Lexical richness ---
    /// Distinct words / total words. Vocabulary diversity; drops as text
    /// length grows, so only comparable across samples of similar length.
    pub type_token_ratio: f32,
    pub avg_word_length: f32,
    /// Fraction of words that occur exactly once — the core statistic behind
    /// Mosteller & Wallace's 1964 Federalist Papers authorship attribution.
    pub hapax_legomena_ratio: f32,

    // --- Syntactic rhythm ---
    pub avg_sentence_length: f32,
    /// Std. dev. of sentence length in words — "burstiness". Alternating
    /// short and long sentences is a real, distinctive rhythmic signature.
    pub sentence_length_stddev: f32,
    pub comma_rate_per_100_words: f32,
    pub semicolon_rate_per_100_words: f32,
    /// Em/en-dash rate — a marker of the aside/appositive habit.
    pub dash_rate_per_100_words: f32,
    pub question_rate_per_100_words: f32,
    pub exclamation_rate_per_100_words: f32,

    // --- Readability (standard closed-form formulas) ---
    /// Flesch (1948) Reading Ease: 0-100, higher = easier.
    pub flesch_reading_ease: f32,
    /// Kincaid et al. (1975) Flesch-Kincaid Grade Level.
    pub flesch_kincaid_grade: f32,
    /// Gunning (1952) Fog Index.
    pub gunning_fog_index: f32,

    // --- Person & address (Pennebaker LIWC pronoun-category tradition) ---
    pub first_person_singular_rate_per_100_words: f32,
    pub first_person_plural_rate_per_100_words: f32,
    pub second_person_rate_per_100_words: f32,
    /// Sentences starting with a bare verb (command mood) / total sentences.
    pub imperative_rate: f32,

    // --- Epistemic stance (Hyland 1998 hedging framework) ---
    /// "definitely", "always", "never", "clearly", ...
    pub certainty_rate_per_100_words: f32,
    /// "maybe", "perhaps", "might", "I think", ...
    pub hedge_rate_per_100_words: f32,

    // --- Cognitive/causal connectives ---
    /// "because", "so", "therefore", "since", ...
    pub causal_connective_rate_per_100_words: f32,
    /// "but", "however", "although", "yet", ...
    pub contrast_connective_rate_per_100_words: f32,

    // --- Deferred: real metrics, but need more than tokenization ---
    /// Heylighen & Dewaele (1999) formality F-score. Needs POS tag counts
    /// (noun/adjective/preposition/article vs. pronoun/verb/adverb/
    /// interjection); `None` until a tagger backend is wired in.
    pub formality_score: Option<f32>,
    /// Lexicon-based polarity (-1..1). `None` until a sourced affect
    /// lexicon (e.g. NRC-VAD) is wired in.
    pub sentiment_polarity: Option<f32>,
    /// Lexicon-based subjectivity (0..1). Same caveat as polarity.
    pub sentiment_subjectivity: Option<f32>,
}

/// Correctly pooled linguistic statistics over many short texts from one
/// author — the corpus-level counterpart to [`LinguisticProfile`].
///
/// [`LinguisticProfile`] is a single-document function. There is no valid way
/// to get a corpus-level equivalent by concatenating many documents and
/// profiling the result once — turns that lack terminal punctuation (the
/// common case for short, informal samples) get fused across document
/// boundaries by the sentence splitter, inflating every sentence-derived
/// field. Averaging many single-document rates is equally invalid wherever
/// the word-count denominator varies from document to document, which it
/// always does. Build one with
/// [`crate::metrics::aggregate_corpus_profile`], which pools every rate as
/// `sum(hits) / sum(words)` across the whole corpus, computes lexical
/// diversity from one corpus-wide word-frequency table rather than
/// per-document unique-word counts (a word appearing once in each of two
/// documents is not a corpus-level hapax), and pools sentence lengths from
/// every document's own, independently computed, sentence boundaries.
///
/// `document_count` and `total_words` travel with every rate so a consumer
/// can see the sample size a statistic rests on, rather than reading a
/// percentage with no denominator.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct CorpusProfile {
    pub document_count: u32,
    pub total_words: u32,

    // --- Lexical richness (corpus-wide word-frequency table) ---
    pub type_token_ratio: f32,
    pub avg_word_length: f32,
    pub hapax_legomena_ratio: f32,

    // --- Syntactic rhythm (sentence lengths pooled per-document, never
    //     from concatenated raw text) ---
    pub avg_sentence_length: f32,
    pub sentence_length_stddev: f32,
    pub comma_rate_per_100_words: f32,
    pub semicolon_rate_per_100_words: f32,
    pub dash_rate_per_100_words: f32,
    pub question_rate_per_100_words: f32,
    pub exclamation_rate_per_100_words: f32,

    // --- Readability ---
    pub flesch_reading_ease: f32,
    pub flesch_kincaid_grade: f32,
    pub gunning_fog_index: f32,

    // --- Person & address ---
    pub first_person_singular_rate_per_100_words: f32,
    pub first_person_plural_rate_per_100_words: f32,
    pub second_person_rate_per_100_words: f32,
    pub imperative_rate: f32,

    // --- Epistemic stance ---
    pub certainty_rate_per_100_words: f32,
    pub hedge_rate_per_100_words: f32,

    // --- Cognitive/causal connectives ---
    pub causal_connective_rate_per_100_words: f32,
    pub contrast_connective_rate_per_100_words: f32,

    // --- Deferred: same caveat as `LinguisticProfile` — measured or `None`,
    //     never approximated ---
    pub formality_score: Option<f32>,
    pub sentiment_polarity: Option<f32>,
    pub sentiment_subjectivity: Option<f32>,
}

/// A single verbatim, attributed utterance — the raw evidence unit of the lexicon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceSample {
    pub id: Uuid,
    /// Multi-author from the start — this crate is not hardcoded to one person.
    pub author: String,
    /// Verbatim text, never altered (typos and all — altering it defeats the point).
    pub text: String,
    pub source: SourceRef,
    pub captured_at: DateTime<Utc>,
    pub word_count: u32,
    pub register: Option<Register>,
    pub tags: Vec<String>,
    pub confidence: Confidence,
    pub profile: LinguisticProfile,
}

/// Category of a distilled `VoicePattern` — the synthesized layer, not raw evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternCategory {
    Opener,
    SentenceConstruction,
    ToneRule,
    Vocabulary,
    AntiPattern,
}

/// A distilled, human-legible rule extracted from a cluster of `VoiceSample`s —
/// the "Tier 1" output an AI actually reads before writing in this voice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoicePattern {
    pub id: Uuid,
    pub author: String,
    pub category: PatternCategory,
    pub description: String,
    /// Pointers back into the evidence (`VoiceSample::id`) this pattern was drawn from.
    pub example_ids: Vec<Uuid>,
    /// `true` = authentic structural voice, safe/desirable to replicate.
    /// `false` = a typing artifact (dropped word, typo) to recognize but never repeat.
    pub replicate: bool,
}
