//! Reference `LexiconEngine` implementations. All gated behind opt-in
//! features so the core crate stays dependency-free by default.

#[cfg(feature = "sqlite-backend")]
mod sqlite;

#[cfg(feature = "sqlite-backend")]
pub use sqlite::SqliteEngine;
