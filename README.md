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
  `patterns_for`.
- **`backend::SqliteEngine`** (opt-in, `sqlite-backend` feature) — a
  reference implementation: SQLite for storage, `fastembed` (pure Rust,
  ONNX, offline after the first model download) for embeddings, brute-force
  cosine similarity via `fastembed::similarity::top_k` for search.

## Why the backend is a feature, not the default

The core (`schema` + `metrics` + `engine` trait) has zero dependencies
beyond `serde`/`uuid`/`chrono`/`thiserror`/`async-trait` by design — this
repo is meant to be genuinely public and auditable, and:

1. Nothing in the core depends on any specific storage/embedding stack, so
   swapping backends never touches the dependency tree consumers of the
   trait actually rely on.
2. Bundling a full embedding stack (ONNX runtime, tokenizers, image codecs
   fastembed pulls in transitively) by default would bloat what should be a
   small core for anyone who only wants the schema and metrics.

`cargo build --features sqlite-backend` opts in to the real thing. Anyone
who'd rather implement `LexiconEngine` against their own storage — Postgres
+ pgvector, a hosted vector DB, whatever they already run — can do that
too, entirely outside this crate's dependency tree.

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

Core (`schema` + `metrics` + `engine` trait) and the `sqlite-backend`
reference implementation are both implemented and tested:
`cargo test --all-features` passes (7/7, including a live ingest→embed→
store→search round trip), `cargo clippy --all-features --all-targets` is
clean on both the default build and the feature-enabled build.

```bash
cargo build                          # core only, no embedding/storage deps
cargo build --features sqlite-backend
```

## License

Dual-licensed under MIT or Apache-2.0, at your option.
