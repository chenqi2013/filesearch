#![cfg_attr(windows, windows_subsystem = "windows")]

mod embedding;
mod extract;
mod indexer;
mod model;
mod search;
mod storage;
mod text_index;
mod watcher;

use axum::extract::{Query, State};
use axum::http::{header, HeaderValue, Method, StatusCode};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use embedding::EmbeddingEngine;
use model::{
    IndexAccepted, IndexFailure, IndexRequest, IndexedChunk, IndexedDocument, Page, PageRequest,
    SearchRequest, SearchResponse, ServiceStats,
};
use parking_lot::{Mutex, RwLock};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};
use std::time::Instant;
use storage::{sqlite_path, Storage};
use text_index::TextIndex;
use tower_http::cors::{AllowOrigin, CorsLayer};
use tower_http::trace::TraceLayer;
use watcher::WatchService;

#[derive(Parser, Debug)]
#[command(version, about = "Local document search core service")]
struct Args {
    #[arg(long, default_value = "127.0.0.1:47653")]
    listen: String,
    #[arg(long, default_value = ".filesearch-data")]
    data_dir: PathBuf,
    #[arg(long)]
    model_dir: Option<PathBuf>,
    #[arg(long, hide = true)]
    extract_file: Option<PathBuf>,
    #[arg(long, hide = true)]
    extract_output: Option<PathBuf>,
}

struct AppState {
    runtime: tokio::runtime::Handle,
    storage: Arc<Storage>,
    text_index: Arc<TextIndex>,
    embedder: Arc<EmbeddingEngine>,
    indexing: AtomicBool,
    processed: AtomicUsize,
    total: AtomicUsize,
    current_file: Mutex<Option<String>>,
    index_timing: Mutex<IndexTimingState>,
    pending_changes: Mutex<HashSet<PathBuf>>,
    watcher_refresh_pending: AtomicBool,
    watcher: Mutex<Option<WatchService>>,
    watcher_status: RwLock<String>,
}

struct IndexTimingState {
    stage: String,
    index_started_at: Option<Instant>,
    stage_started_at: Option<Instant>,
    total_elapsed_ms: u128,
    scan_ms: u128,
    check_ms: u128,
    parse_ms: u128,
    embedding_ms: u128,
    storage_ms: u128,
    text_index_ms: u128,
}

impl Default for IndexTimingState {
    fn default() -> Self {
        Self {
            stage: "ready".to_owned(),
            index_started_at: None,
            stage_started_at: None,
            total_elapsed_ms: 0,
            scan_ms: 0,
            check_ms: 0,
            parse_ms: 0,
            embedding_ms: 0,
            storage_ms: 0,
            text_index_ms: 0,
        }
    }
}

struct IndexTimingSnapshot {
    stage: String,
    stage_elapsed_ms: u128,
    total_elapsed_ms: u128,
    scan_ms: u128,
    check_ms: u128,
    parse_ms: u128,
    embedding_ms: u128,
    storage_ms: u128,
    text_index_ms: u128,
}

type SharedState = Arc<AppState>;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let args = Args::parse();
    if let Some(input) = args.extract_file.as_deref() {
        let output = args
            .extract_output
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("缺少提取结果输出路径"))?;
        let text = extract::extract_text(input)?;
        std::fs::write(output, text.as_bytes())?;
        return Ok(());
    }
    std::fs::create_dir_all(&args.data_dir)?;

    let storage = Arc::new(Storage::open(&sqlite_path(&args.data_dir))?);
    let migrated = storage.migrate_legacy(&args.data_dir.join("index.json"))?;
    let text_index = Arc::new(TextIndex::open(&args.data_dir.join("tantivy"))?);
    let counts = storage.counts()?;
    if text_index.document_count() != counts.chunks as u64 {
        tracing::info!(
            sqlite_chunks = counts.chunks,
            "rebuilding Tantivy index from SQLite"
        );
        text_index.rebuild(&storage.list_documents()?, &storage.list_chunks()?)?;
    }
    let model_dir = args.model_dir.unwrap_or_else(|| {
        std::env::current_exe()
            .ok()
            .and_then(|path| path.parent().map(PathBuf::from))
            .map(|path| path.join("models").join("embedding-rwkv-tiny"))
            .unwrap_or_else(|| args.data_dir.join("models").join("embedding-rwkv-tiny"))
    });
    let embedder = Arc::new(EmbeddingEngine::new(model_dir));
    let state = Arc::new(AppState {
        runtime: tokio::runtime::Handle::current(),
        storage,
        text_index,
        embedder,
        indexing: AtomicBool::new(false),
        processed: AtomicUsize::new(0),
        total: AtomicUsize::new(0),
        current_file: Mutex::new(None),
        index_timing: Mutex::new(IndexTimingState::default()),
        pending_changes: Mutex::new(HashSet::new()),
        watcher_refresh_pending: AtomicBool::new(false),
        watcher: Mutex::new(None),
        watcher_status: RwLock::new("starting".to_owned()),
    });
    restart_watcher(&state);

    let directories = state.storage.directories()?;
    let profile_changed = !state.storage.embedding_profile_matches()?;
    if profile_changed && !directories.is_empty() {
        state.storage.clear_embeddings()?;
    }
    if (migrated || profile_changed || state.storage.missing_embedding_count()? > 0)
        && !directories.is_empty()
    {
        launch_full_index(Arc::clone(&state), directories);
    }

    let app = Router::new()
        .route("/health", get(health))
        .route("/stats", get(stats))
        .route("/documents", get(documents))
        .route("/chunks", get(chunks))
        .route("/search", post(search_documents))
        .route("/index", post(start_index))
        .route("/failures", get(failures))
        .layer(
            CorsLayer::new()
                .allow_origin(AllowOrigin::list([
                    HeaderValue::from_static("http://127.0.0.1:1420"),
                    HeaderValue::from_static("http://localhost:1420"),
                    HeaderValue::from_static("tauri://localhost"),
                    HeaderValue::from_static("http://tauri.localhost"),
                ]))
                .allow_methods([Method::GET, Method::POST])
                .allow_headers([header::CONTENT_TYPE]),
        )
        .layer(TraceLayer::new_for_http())
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(&args.listen).await?;
    tracing::info!(address = %args.listen, "search core listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn health(State(state): State<SharedState>) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "service": "search-core",
        "version": env!("CARGO_PKG_VERSION"),
        "storage": "sqlite+tantivy",
        "embedding": state.embedder.status(),
        "embedding_backend": state.embedder.backend(),
    }))
}

async fn stats(State(state): State<SharedState>) -> Json<ServiceStats> {
    let counts = state.storage.counts().unwrap_or(storage::StorageCounts {
        documents: 0,
        chunks: 0,
        failures: 0,
    });
    let indexing = state.indexing.load(Ordering::Relaxed);
    let timing = index_timing_snapshot(&state, indexing);
    Json(ServiceStats {
        status: if indexing { "indexing" } else { "ready" }.to_owned(),
        document_count: counts.documents,
        chunk_count: counts.chunks,
        failed_count: counts.failures,
        processed_files: state.processed.load(Ordering::Relaxed),
        total_files: state.total.load(Ordering::Relaxed),
        current_file: state.current_file.lock().clone(),
        index_stage: timing.stage,
        index_stage_elapsed_ms: timing.stage_elapsed_ms,
        index_total_elapsed_ms: timing.total_elapsed_ms,
        index_scan_ms: timing.scan_ms,
        index_check_ms: timing.check_ms,
        index_parse_ms: timing.parse_ms,
        index_embedding_ms: timing.embedding_ms,
        index_storage_ms: timing.storage_ms,
        index_text_index_ms: timing.text_index_ms,
        directories: state.storage.directories().unwrap_or_default(),
        last_indexed: state.storage.last_indexed().unwrap_or_default(),
        storage_backend: "SQLite WAL + Tantivy BM25".to_owned(),
        embedding_model: state.embedder.status(),
        embedding_backend: state.embedder.backend(),
        watcher_status: state.watcher_status.read().clone(),
    })
}

async fn documents(
    State(state): State<SharedState>,
    Query(request): Query<PageRequest>,
) -> Result<Json<Page<IndexedDocument>>, (StatusCode, Json<Value>)> {
    let request = request.bounded();
    let counts = state.storage.counts().map_err(internal_error)?;
    let items = state
        .storage
        .document_page(request.offset, request.limit)
        .map_err(internal_error)?;
    Ok(Json(Page {
        items,
        total: counts.documents,
        offset: request.offset,
        limit: request.limit,
    }))
}

async fn chunks(
    State(state): State<SharedState>,
    Query(request): Query<PageRequest>,
) -> Result<Json<Page<IndexedChunk>>, (StatusCode, Json<Value>)> {
    let request = request.bounded();
    let counts = state.storage.counts().map_err(internal_error)?;
    let items = state
        .storage
        .chunk_page(request.offset, request.limit)
        .map_err(internal_error)?;
    Ok(Json(Page {
        items,
        total: counts.chunks,
        offset: request.offset,
        limit: request.limit,
    }))
}

async fn failures(
    State(state): State<SharedState>,
    Query(request): Query<PageRequest>,
) -> Result<Json<Page<IndexFailure>>, (StatusCode, Json<Value>)> {
    let request = request.bounded();
    let counts = state.storage.counts().map_err(internal_error)?;
    let items = state
        .storage
        .failure_page(request.offset, request.limit)
        .map_err(internal_error)?;
    Ok(Json(Page {
        items,
        total: counts.failures,
        offset: request.offset,
        limit: request.limit,
    }))
}

async fn search_documents(
    State(state): State<SharedState>,
    Json(request): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, Json<Value>)> {
    if request.query.trim().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "搜索内容不能为空" })),
        ));
    }
    let started = Instant::now();
    let query = request.query.clone();
    let task_state = Arc::clone(&state);
    let results = tokio::task::spawn_blocking(move || {
        search::search(
            &task_state.storage,
            &task_state.text_index,
            &task_state.embedder,
            &request,
        )
    })
    .await
    .map_err(internal_error)?
    .map_err(internal_error)?;
    Ok(Json(SearchResponse {
        query,
        total: results.len(),
        elapsed_ms: started.elapsed().as_millis(),
        results,
    }))
}

async fn start_index(
    State(state): State<SharedState>,
    Json(request): Json<IndexRequest>,
) -> Result<(StatusCode, Json<IndexAccepted>), (StatusCode, Json<Value>)> {
    if !launch_full_index(Arc::clone(&state), request.paths) {
        return Ok((
            StatusCode::CONFLICT,
            Json(IndexAccepted {
                accepted: false,
                message: "索引任务正在运行".to_owned(),
            }),
        ));
    }
    Ok((
        StatusCode::ACCEPTED,
        Json(IndexAccepted {
            accepted: true,
            message: "索引任务已启动".to_owned(),
        }),
    ))
}

fn launch_full_index(state: SharedState, paths: Vec<String>) -> bool {
    if state
        .indexing
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return false;
    }
    state.processed.store(0, Ordering::Relaxed);
    state.total.store(0, Ordering::Relaxed);
    reset_index_timing(&state);
    let runtime = state.runtime.clone();
    runtime.spawn_blocking(move || {
        let result = indexer::build_index(
            &paths,
            &state.storage,
            &state.text_index,
            &state.embedder,
            |update| update_progress(&state, update),
        );
        if let Err(error) = result {
            tracing::error!(%error, "full index task failed");
            let _ = state.storage.record_failure(&IndexFailure {
                path: "<index-task>".to_owned(),
                category: "internal".to_owned(),
                reason: format!("索引任务失败: {error:#}"),
            });
        }
        finish_index_task(&state, true);
    });
    true
}

fn queue_incremental(state: SharedState, paths: Vec<PathBuf>) {
    state.pending_changes.lock().extend(paths);
    launch_pending_incremental(state);
}

fn launch_pending_incremental(state: SharedState) {
    if state.pending_changes.lock().is_empty()
        || state
            .indexing
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
    {
        return;
    }
    reset_index_timing(&state);
    let runtime = state.runtime.clone();
    runtime.spawn_blocking(move || {
        loop {
            let paths = state.pending_changes.lock().drain().collect::<Vec<_>>();
            if paths.is_empty() {
                break;
            }
            if let Err(error) = indexer::update_paths(
                &paths,
                &state.storage,
                &state.text_index,
                &state.embedder,
                |update| update_progress(&state, update),
            ) {
                tracing::error!(%error, "incremental index task failed");
            }
        }
        finish_index_task(&state, false);
    });
}

fn finish_index_task(state: &SharedState, refresh_watcher: bool) {
    finish_index_timing(state);
    state
        .processed
        .store(state.total.load(Ordering::Relaxed), Ordering::Relaxed);
    *state.current_file.lock() = None;
    state.indexing.store(false, Ordering::SeqCst);
    if refresh_watcher || state.watcher_refresh_pending.swap(false, Ordering::SeqCst) {
        restart_watcher(state);
    }
    if !state.pending_changes.lock().is_empty() {
        launch_pending_incremental(Arc::clone(state));
    }
}

fn update_progress(state: &AppState, update: indexer::ProgressUpdate<'_>) {
    state.processed.store(update.processed, Ordering::Relaxed);
    state.total.store(update.total, Ordering::Relaxed);
    *state.current_file.lock() = update.current_file.map(ToOwned::to_owned);
    let mut timing = state.index_timing.lock();
    if timing.stage != update.stage {
        timing.stage = update.stage.to_owned();
        timing.stage_started_at = Some(Instant::now());
    }
    timing.scan_ms = update.scan_ms;
    timing.check_ms = update.check_ms;
    timing.parse_ms = update.parse_ms;
    timing.embedding_ms = update.embedding_ms;
    timing.storage_ms = update.storage_ms;
    timing.text_index_ms = update.text_index_ms;
}

fn reset_index_timing(state: &AppState) {
    let now = Instant::now();
    *state.index_timing.lock() = IndexTimingState {
        stage: "starting".to_owned(),
        index_started_at: Some(now),
        stage_started_at: Some(now),
        ..IndexTimingState::default()
    };
}

fn finish_index_timing(state: &AppState) {
    let mut timing = state.index_timing.lock();
    timing.total_elapsed_ms = timing
        .index_started_at
        .map(|started| started.elapsed().as_millis())
        .unwrap_or(timing.total_elapsed_ms);
    timing.stage = "ready".to_owned();
    timing.index_started_at = None;
    timing.stage_started_at = None;
}

fn index_timing_snapshot(state: &AppState, indexing: bool) -> IndexTimingSnapshot {
    let timing = state.index_timing.lock();
    IndexTimingSnapshot {
        stage: timing.stage.clone(),
        stage_elapsed_ms: if indexing {
            timing
                .stage_started_at
                .map(|started| started.elapsed().as_millis())
                .unwrap_or(0)
        } else {
            0
        },
        total_elapsed_ms: if indexing {
            timing
                .index_started_at
                .map(|started| started.elapsed().as_millis())
                .unwrap_or(timing.total_elapsed_ms)
        } else {
            timing.total_elapsed_ms
        },
        scan_ms: timing.scan_ms,
        check_ms: timing.check_ms,
        parse_ms: timing.parse_ms,
        embedding_ms: timing.embedding_ms,
        storage_ms: timing.storage_ms,
        text_index_ms: timing.text_index_ms,
    }
}

fn restart_watcher(state: &SharedState) {
    let roots = state
        .storage
        .directories()
        .unwrap_or_default()
        .into_iter()
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    let paths_state: Weak<AppState> = Arc::downgrade(state);
    let error_state: Weak<AppState> = Arc::downgrade(state);
    let recovered_state: Weak<AppState> = Arc::downgrade(state);
    match WatchService::start(
        roots,
        move |paths| {
            if let Some(state) = paths_state.upgrade() {
                queue_incremental(state, paths);
            }
        },
        move |message| {
            if let Some(state) = error_state.upgrade() {
                *state.watcher_status.write() = format!("degraded: {message}");
                let _ = state.storage.record_failure(&IndexFailure {
                    path: "<file-watcher>".to_owned(),
                    category: "offline".to_owned(),
                    reason: message,
                });
            }
        },
        move |path| {
            if let Some(state) = recovered_state.upgrade() {
                state.watcher_refresh_pending.store(true, Ordering::SeqCst);
                *state.watcher_status.write() = "recovering".to_owned();
                queue_incremental(state, vec![path]);
            }
        },
    ) {
        Ok(watcher) => {
            *state.watcher_status.write() = watcher.status();
            *state.watcher.lock() = Some(watcher);
        }
        Err(error) => {
            *state.watcher_status.write() = format!("error: {error}");
            tracing::error!(%error, "failed to start file watcher");
        }
    }
}

fn internal_error(error: impl std::fmt::Display) -> (StatusCode, Json<Value>) {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({ "error": error.to_string() })),
    )
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler")
    };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! { _ = ctrl_c => {}, _ = terminate => {} }
}
