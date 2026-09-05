# larc-lexicon-engine

A schema and pure-Rust metrics engine for building queryable **writing-voice
lexicons** from real authored text — the evidence base an LLM can condition
on to write *in* someone's actual voice, instead of a generic paraphrase of it.

## What this is

Three pieces, deliberately separated:

- **`schema`** — `VoiceSample` (one verbatim, attributed, feature-scored
  utterance) and `VoicePattern` (a distilled, human-legible rule drawn from a
  cluster of samples — the thing an AI actually reads before writing).
- **`metrics`** — `compute_linguistic_profile(text) -> LinguisticProfile`.
  Every field is computed from tokenization and word-list lookups alone —
  no POS tagger, no trained model, no network call. Deterministic and
  reproducible from the text itself.
- **`engine`** — the `LexiconEngine` trait: `ingest` / `search` /
  `patterns_for`. No concrete backend ships in this crate (see below).

## Why no backend ships here

The obvious backend is a local embedding model + a small vector store — but
this repo is meant to be genuinely public, and:

1. A path-dependency on any specific private vector-store crate breaks the
   build for anyone else who clones this.
2. Bundling a full embedding stack (model weights or an ONNX runtime) bloats
   what should be a small, auditable core.

So the core crate ships the trait, not an implementation. Implementing
`LexiconEngine` against your own storage — SQLite + local embeddings,
Postgres + pgvector, a hosted vector DB, whatever you already run — is a
small amount of glue code outside this crate, and it never has to touch
this repository's dependency tree.

## The `LinguisticProfile` feature set — and why these fields specifically

The goal was a feature set "as robust as possible" while pruning noise: a
full LIWC-style run has 80+ categories, most of which (social/religion/
leisure/money) are corpus-linguistics research tools with no bearing on
*how someone writes*. What's kept here is the subset that's (a) established
in the stylometry/computational-linguistics literature, (b) computable
without a POS tagger or trained model, and (c) actually useful for
conditioning an LLM's writing style or filtering retrieval results.

| Category | Fields | Basis |
|---|---|---|
| Lexical richness | type-token ratio, avg word length, hapax legomena ratio | Mosteller & Wallace (1964) — the Federalist Papers authorship-attribution method; hapax rate is still a stylometry baseline |
| Syntactic rhythm | avg/stddev sentence length, comma/semicolon/dash/question/exclamation rate | Sentence-length "burstiness" (alternating short/long) is a well-documented individual fingerprint |
| Readability | Flesch Reading Ease, Flesch-Kincaid Grade, Gunning Fog | Flesch (1948); Kincaid et al. (1975); Gunning (1952) — closed-form, no ML needed |
| Person & address | 1st-singular/1st-plural/2nd-person rate, imperative rate | Pennebaker's LIWC pronoun-category tradition — distinguishes reflective vs. directive voice |
| Epistemic stance | certainty rate, hedge rate | Hyland (1998) hedging framework — "this is broken" vs. "this might be an issue" |
| Cognitive connectives | causal rate, contrastive rate | Flags reasoning style (chained justification vs. corrective pivoting) |

Two fields are real, cited metrics but deliberately left `Option` and
`None` rather than faked:

- **`formality_score`** — Heylighen & Dewaele's (1999) F-score needs POS
  tag counts (noun/adjective/preposition/article vs. pronoun/verb/adverb/
  interjection). No tagger is bundled; wire one in and populate this field
  rather than approximating it with tokenization.
- **`sentiment_polarity` / `sentiment_subjectivity`** — needs a sourced
  affect lexicon (e.g. NRC-VAD). Same rule: measured or `None`, never guessed.

Dropped entirely as noise for this use case: full LIWC's 80+ categories,
POS n-gram models (needs a real parser — disproportionate for a personal
lexicon), and any sentiment score without a defensible lexicon behind it.

## Trust tiering

`Confidence::{Verbatim, Relayed, Paraphrased}` on every sample. A lexicon
built on paraphrased text learns the paraphraser's voice, not the author's
— this field exists so a consumer can filter down to `Verbatim` before
synthesizing style rules, while still keeping lower-trust samples around
for topical search.

## Status

Core (`schema` + `metrics` + `engine` trait) is implemented and tested —
`cargo test` passes, `cargo clippy` is clean. No backend implementation
exists yet; that's real, separate work, not stubbed out here.

## License

Dual-licensed under MIT or Apache-2.0, at your option.
