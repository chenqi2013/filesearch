use crate::embedding::{lexical_text, query_terms, ASCII_PREFIX_MIN_LENGTH};
use crate::model::{StoredChunk, StoredDocument};
use anyhow::{Context, Result};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::path::Path;
use tantivy::collector::TopDocs;
use tantivy::query::{BooleanQuery, FuzzyTermQuery, Occur, Query, TermQuery};
use tantivy::schema::{Field, IndexRecordOption, Schema, Value, INDEXED, STORED, STRING, TEXT};
use tantivy::{doc, Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument, Term};

const WRITER_MEMORY_BYTES: usize = 128 * 1024 * 1024;
const REBUILD_COMMIT_BATCH_SIZE: usize = 2_000;

#[derive(Debug, Clone)]
pub struct KeywordHit {
    pub chunk_id: u64,
    pub score: f32,
}

pub struct TextIndex {
    index: Index,
    reader: IndexReader,
    writer: Mutex<IndexWriter>,
    content: Field,
    chunk_id: Field,
    document_id: Field,
}

impl TextIndex {
    pub fn open(path: &Path) -> Result<Self> {
        std::fs::create_dir_all(path)?;
        let schema = build_schema();
        let index = if path.join("meta.json").exists() {
            Index::open_in_dir(path).context("无法打开 Tantivy 索引")?
        } else {
            Index::create_in_dir(path, schema).context("无法创建 Tantivy 索引")?
        };
        let schema = index.schema();
        let content = schema.get_field("content")?;
        let chunk_id = schema.get_field("chunk_id")?;
        let document_id = schema.get_field("document_id")?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;
        let writer = index.writer_with_num_threads(1, WRITER_MEMORY_BYTES)?;
        Ok(Self {
            index,
            reader,
            writer: Mutex::new(writer),
            content,
            chunk_id,
            document_id,
        })
    }

    #[cfg(test)]
    pub fn replace_document(
        &self,
        document_id: &str,
        name: &str,
        chunks: &[StoredChunk],
    ) -> Result<()> {
        self.replace_documents(&[(document_id, name, chunks)])
    }

    pub fn replace_documents(&self, documents: &[(&str, &str, &[StoredChunk])]) -> Result<()> {
        let writer = self.writer.lock();
        for (document_id, name, chunks) in documents {
            writer.delete_term(Term::from_field_text(self.document_id, document_id));
            for chunk in *chunks {
                let mut searchable = lexical_text(name);
                if !searchable.is_empty() {
                    searchable = format!("{searchable} {searchable} {searchable}");
                }
                let body = lexical_text(&chunk.text);
                if !body.is_empty() {
                    if !searchable.is_empty() {
                        searchable.push(' ');
                    }
                    searchable.push_str(&body);
                }
                writer.add_document(doc!(
                    self.chunk_id => chunk.id,
                    self.document_id => *document_id,
                    self.content => searchable,
                ))?;
            }
        }
        Ok(())
    }

    pub fn delete_document(&self, document_id: &str) {
        self.writer
            .lock()
            .delete_term(Term::from_field_text(self.document_id, document_id));
    }

    pub fn commit(&self) -> Result<()> {
        self.writer.lock().commit()?;
        self.reader.reload()?;
        Ok(())
    }

    pub fn rebuild(&self, documents: &[StoredDocument], chunks: &[StoredChunk]) -> Result<()> {
        let names = documents
            .iter()
            .map(|document| (document.id.as_str(), document.name.as_str()))
            .collect::<HashMap<_, _>>();
        let mut writer = self.writer.lock();
        writer.delete_all_documents()?;
        writer.commit()?;
        for (index, chunk) in chunks.iter().enumerate() {
            let name = names
                .get(chunk.document_id.as_str())
                .copied()
                .unwrap_or_default();
            let mut searchable = lexical_text(name);
            if !searchable.is_empty() {
                searchable = format!("{searchable} {searchable} {searchable}");
            }
            let body = lexical_text(&chunk.text);
            if !body.is_empty() {
                if !searchable.is_empty() {
                    searchable.push(' ');
                }
                searchable.push_str(&body);
            }
            writer.add_document(doc!(
                self.chunk_id => chunk.id,
                self.document_id => chunk.document_id.as_str(),
                self.content => searchable,
            ))?;
            if (index + 1) % REBUILD_COMMIT_BATCH_SIZE == 0 {
                writer.commit()?;
            }
        }
        writer.commit()?;
        drop(writer);
        self.reader.reload()?;
        Ok(())
    }

    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<KeywordHit>> {
        let terms = query_terms(query);
        if terms.is_empty() {
            return Ok(Vec::new());
        }
        let parsed = keyword_query(&terms, self.content);
        let searcher = self.reader.searcher();
        let top_docs = searcher.search(&parsed, &TopDocs::with_limit(limit.clamp(1, 2_000)))?;
        let mut hits = Vec::with_capacity(top_docs.len());
        for (score, address) in top_docs {
            let document = searcher.doc::<TantivyDocument>(address)?;
            if let Some(chunk_id) = document
                .get_first(self.chunk_id)
                .and_then(|value| value.as_u64())
            {
                hits.push(KeywordHit { chunk_id, score });
            }
        }
        Ok(hits)
    }

    pub fn document_count(&self) -> u64 {
        self.reader.searcher().num_docs()
    }
}

fn keyword_query(terms: &[String], content: Field) -> Box<dyn Query> {
    let clauses = terms
        .iter()
        .map(|term| {
            let term = Term::from_field_text(content, term);
            if term
                .value()
                .as_str()
                .is_some_and(|value| value.len() >= ASCII_PREFIX_MIN_LENGTH)
            {
                (
                    Occur::Should,
                    Box::new(FuzzyTermQuery::new_prefix(term, 0, true)) as Box<dyn Query>,
                )
            } else {
                (
                    Occur::Should,
                    Box::new(TermQuery::new(term, IndexRecordOption::WithFreqs)) as Box<dyn Query>,
                )
            }
        })
        .collect();
    Box::new(BooleanQuery::new(clauses))
}

fn build_schema() -> Schema {
    let mut builder = Schema::builder();
    builder.add_u64_field("chunk_id", INDEXED | STORED);
    builder.add_text_field("document_id", STRING | STORED);
    builder.add_text_field("content", TEXT);
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tantivy_matches_chinese_bigrams() {
        let directory = tempfile::tempdir().unwrap();
        let index = TextIndex::open(directory.path()).unwrap();
        index
            .replace_document(
                "doc-1",
                "Windows 本地文档搜索需求",
                &[StoredChunk {
                    id: 1,
                    document_id: "doc-1".to_owned(),
                    text: "支持本地语义检索和关键词查询".to_owned(),
                }],
            )
            .unwrap();
        index.commit().unwrap();
        let hits = index.search("本地搜索", 10).unwrap();
        assert_eq!(hits.first().map(|hit| hit.chunk_id), Some(1));
        assert!(index.search("如何制作搜索引擎蛋糕", 10).is_ok());
        assert!(index.search("本年度天气预报", 10).unwrap().is_empty());
        assert!(!index.search("本", 10).unwrap().is_empty());
    }

    #[test]
    fn tantivy_matches_ascii_prefixes() {
        let directory = tempfile::tempdir().unwrap();
        let index = TextIndex::open(directory.path()).unwrap();
        index
            .replace_document(
                "doc-1",
                "11.xlsx",
                &[StoredChunk {
                    id: 1,
                    document_id: "doc-1".to_owned(),
                    text: "Pellentesque sit amet lectus. Praesent pulvinar, nunc quis iaculis sagittis".to_owned(),
                }],
            )
            .unwrap();
        index.commit().unwrap();
        assert_eq!(
            index
                .search("Praesent pulv", 10)
                .unwrap()
                .first()
                .map(|hit| hit.chunk_id),
            Some(1)
        );
        assert_eq!(
            index
                .search("Praesent pu", 10)
                .unwrap()
                .first()
                .map(|hit| hit.chunk_id),
            Some(1)
        );
        assert_eq!(
            index
                .search("pulv", 10)
                .unwrap()
                .first()
                .map(|hit| hit.chunk_id),
            Some(1)
        );
    }
}
