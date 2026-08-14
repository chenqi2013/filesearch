mod extract;
mod indexer;
mod model;
mod search;

use axum::extract::State;
use axum::http::{header, HeaderValue, Method, StatusCode};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use indexer::{build_index, load, save};
use model::{
    IndexAccepted, IndexRequest, PersistedIndex, SearchRequest, SearchResponse, ServiceStats,
};
use parking_lot::{Mutex, RwLock};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tower_http::cors::{AllowOrigin, CorsLayer};
use tower_http::trace::TraceLayer;

#[derive(Parser, Debug)]
#[command(version, about = "Local document search core service")]
struct Args {
    #[arg(long, default_value = "127.0.0.1:47653")]
    listen: String,
    #[arg(long, default_value = ".filesearch-data")]
    data_dir: PathBuf,
}

struct AppState {
    index: RwLock<PersistedIndex>,
    index_path: PathBuf,
    indexing: AtomicBool,
    processed: AtomicUsize,
    total: AtomicUsize,
    current_file: Mutex<Option<String>>,
}

type SharedState = Arc<AppState>;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let args = Args::parse();
    let index_path = args.data_dir.join("index.json");
    let state = Arc::new(AppState {
        index: RwLock::new(load(&index_path)),
        index_path,
        indexing: AtomicBool::new(false),
        processed: AtomicUsize::new(0),
        total: AtomicUsize::new(0),
        current_file: Mutex::new(None),
    });
    let app = Router::new()
        .route("/health", get(health))
        .route("/stats", get(stats))
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

async fn health() -> Json<Value> {
    Json(json!({ "status": "ok", "service": "search-core", "version": env!("CARGO_PKG_VERSION") }))
}

async fn stats(State(state): State<SharedState>) -> Json<ServiceStats> {
    let index = state.index.read();
    Json(ServiceStats {
        status: if state.indexing.load(Ordering::Relaxed) {
            "indexing".to_owned()
        } else {
            "ready".to_owned()
        },
        document_count: index.documents.len(),
        chunk_count: index
            .documents
            .iter()
            .map(|document| document.chunks.len())
            .sum(),
        failed_count: index.failures.len(),
        processed_files: state.processed.load(Ordering::Relaxed),
        total_files: state.total.load(Ordering::Relaxed),
        current_file: state.current_file.lock().clone(),
        directories: index.directories.clone(),
        last_indexed: index.last_indexed.clone(),
    })
}

async fn failures(State(state): State<SharedState>) -> Json<Value> {
    Json(json!({ "failures": state.index.read().failures.clone() }))
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
    let index = state.index.read();
    let results = search::search(&index.documents, &request);
    Ok(Json(SearchResponse {
        query: request.query,
        total: results.len(),
        elapsed_ms: started.elapsed().as_millis(),
        results,
    }))
}

async fn start_index(
    State(state): State<SharedState>,
    Json(request): Json<IndexRequest>,
) -> Result<(StatusCode, Json<IndexAccepted>), (StatusCode, Json<Value>)> {
    if state
        .indexing
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return Ok((
            StatusCode::CONFLICT,
            Json(IndexAccepted {
                accepted: false,
                message: "索引任务正在运行".to_owned(),
            }),
        ));
    }

    let task_state = Arc::clone(&state);
    tokio::task::spawn_blocking(move || {
        let previous = task_state.index.read().clone();
        let next = build_index(&request.paths, &previous, |update| {
            task_state
                .processed
                .store(update.processed, Ordering::Relaxed);
            task_state.total.store(update.total, Ordering::Relaxed);
            *task_state.current_file.lock() = update.current_file.map(ToOwned::to_owned);
        });
        if let Err(error) = save(&task_state.index_path, &next) {
            tracing::error!(%error, "failed to persist index");
        }
        *task_state.index.write() = next;
        *task_state.current_file.lock() = None;
        task_state.indexing.store(false, Ordering::SeqCst);
    });

    Ok((
        StatusCode::ACCEPTED,
        Json(IndexAccepted {
            accepted: true,
            message: "索引任务已启动".to_owned(),
        }),
    ))
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
