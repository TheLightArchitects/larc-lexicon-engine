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

## The `larc` CLI

```bash
cargo install --path . --features cli,sqlite-backend
```

| Command | What it does | Needs |
|---|---|---|
| `larc profile <file>` | Print the `LinguisticProfile` of a file (`-` for stdin). `--json` for the raw struct. | `cli` |
| `larc ingest file <path> --author X` | Store one file as a single sample. `--confidence`, `--source-kind`, `--register`, `--tags`, `--project`. | `cli,sqlite-backend` |
| `larc ingest claude-sessions <dir> --author X` | Store every human-authored turn from a directory of session transcripts. `--min-words`, `--dry-run`. | `cli,sqlite-backend` |
| `larc search <query> [--author X] [--top-k N]` | Embedding similarity search over stored samples. | `cli,sqlite-backend` |
| `larc stats [author] [--tag T] [--json]` | Correctly pooled `CorpusProfile` over stored samples — the whole population, not a `search` subset. | `cli,sqlite-backend` |
| `larc distill <author> [--tag T] [--dry-run]` | Derive candidate `VoicePattern`s from measured evidence (opener habits, punctuation, hedging, connective skew) instead of writing them by hand — see below. | `cli,sqlite-backend` |
| `larc patterns list <author>` / `larc patterns add …` | Read and write distilled `VoicePattern`s. | `cli,sqlite-backend` |

The `cli` feature alone builds only `larc profile`, which needs nothing but the
core crate. Commands that touch a lexicon are compiled in by `sqlite-backend`,
rather than appearing in `--help` and failing at runtime.

The database lives at `--db`, else `$LARC_LEXICON_DB`, else
`~/.larc-lexicon/voice.db`.

Sample ids are UUID v5 over origin + exact text (for session ingest, origin
includes the turn's position in its session, so two identical turns — two
separate "yes" confirmations — don't collide), so re-running an ingest as a
corpus grows **upserts instead of duplicating**, and an edited turn becomes a
new sample rather than a silent overwrite.

### Why transcript ingest is stricter than it looks

A Claude Code transcript stores several different things under the same
`"type": "user"` tag. Across 14,425 such entries in one real nine-session
project:

| `origin.kind` | `content` | count | what it is |
|---|---|---:|---|
| absent | array | 11,537 | `tool_result` blocks re-injected as user turns |
| absent | string | 2,226 | harness/hook-injected synthetic turns |
| `task-notification` | string | 342 | background-task completion notices |
| `human` | string | 304 | **the author's typed turns** |
| `human` | array | 10 | **the author's text, plus an attachment** |
| `peer` | string | 6 | messages relayed *from another agent session* |

The obvious filter — `type == "user"` with a string body — captures 2,878
entries, of which **2,574 (89%) are not the author's writing at all**. It would
attribute harness boilerplate and other agents' prose to the human: exactly the
contamination `Confidence` exists to prevent, arriving through the ingest path
instead. So authorship here is decided by `origin.kind == "human"` alone.

Body *shape* is then a recall question rather than an authorship one. Once an
entry is known to be human-authored, an array body is not a tool result — those
never carry a `human` origin — but a message with an attachment, and its `text`
blocks are ordinary prose, often the most opinionated kind since they react to a
screenshot. Those are kept; only the ones whose entire text is an `[Image #N]`
marker drop out, having no words to measure.

Even a correctly identified turn is not clean text: the harness appends
`<system-reminder>` blocks and slash-command echoes to what the human typed.
Those are stripped before profiling, or every metric is computed partly over
boilerplate nobody wrote.

### `larc distill` — measured patterns, not hand-written ones

A `VoicePattern` written by hand has two structural weaknesses: nothing links
its description back to the specific samples that support it, and re-running
the same analysis by hand after the corpus grows has no guarantee it's
computed the same way twice. `larc distill` fixes both by scanning every
sample's text individually (never by concatenating them — the same
turn-fusion mistake `CorpusProfile` exists to avoid) for a fixed set of
discrete markers — acknowledgment openers, bare-imperative openers, missing
terminal punctuation, gratitude markers, hedge presence, and a connective-skew
check read straight from `aggregate_corpus_profile`'s pooled rates. Each
marker becomes a pattern only once it clears both a minimum sample size and a
minimum effect size, and every emitted pattern cites the exact count behind
it plus up to three real `example_ids`.

Pattern ids are deterministic per `(author, signal)`, so re-running `distill`
as a corpus grows upserts each signal's pattern with fresher numbers rather
than accumulating duplicates. It will also tell you when your data is dirty:
on one real corpus, `distill` reported a 13:1 causal-over-contrastive
connective skew — until three pasted documents identified by the paste guard
above were removed from storage, after which the same command reported no
skew at all (the true ratio was ~2:1, under the threshold). The pattern
wasn't wrong given the input; the input was wrong, and a measured pipeline
surfaces that instead of hiding it the way a one-off hand analysis would.

## Status

Core, the `sqlite-backend` reference implementation, and the `larc` CLI are
implemented and tested. `cargo test --all-features` passes 45/45 — including a
live ingest→embed→store→search round trip, idempotent re-ingest, the
pattern write/read round trip, corpus-aggregation correctness (pooled rates,
sentence-boundary handling, lexical-diversity scoping), and pattern
distillation (threshold gating, deterministic ids, evidence linkage) — and
`cargo clippy --all-targets -- -D warnings` is clean on the default, `cli`,
and `--all-features` builds.

```bash
cargo build                                   # core only, no embedding/storage deps
cargo build --features sqlite-backend
cargo build --features cli                    # `larc profile` only
cargo build --features cli,sqlite-backend     # the full CLI
```

## License

Dual-licensed under MIT or Apache-2.0, at your option.
