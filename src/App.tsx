import { FormEvent, Fragment, useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import {
  AlertTriangle,
  ChevronLeft,
  ChevronRight,
  Clock3,
  Cpu,
  ExternalLink,
  File,
  FileSearch,
  Folder,
  FolderOpen,
  Globe2,
  Plus,
  RefreshCw,
  Search,
  Trash2,
  X,
} from "lucide-react";
import { coreApi } from "./api";
import { getMessages, type Locale } from "./i18n";
import type {
  IndexFailure,
  IndexedChunk,
  IndexedDocument,
  Page,
  SearchMode,
  SearchResponse,
  SearchResult,
  ServiceStats,
} from "./types";

type InventoryKind = "documents" | "chunks" | "failures";
type InventoryState =
  | { kind: "documents"; page: Page<IndexedDocument> }
  | { kind: "chunks"; page: Page<IndexedChunk> }
  | { kind: "failures"; page: Page<IndexFailure> };

const EMPTY_STATS: ServiceStats = {
  status: "ready",
  document_count: 0,
  chunk_count: 0,
  failed_count: 0,
  processed_files: 0,
  total_files: 0,
  directories: [],
  extensions: [],
  embedding_backend: "cpu",
};

const isTauri = () => "__TAURI_INTERNALS__" in window;

function App() {
  const [locale, setLocale] = useState<Locale>("zh");
  const t = getMessages(locale);
  const [stats, setStats] = useState<ServiceStats>(EMPTY_STATS);
  const [connected, setConnected] = useState(false);
  const [query, setQuery] = useState("");
  const [submittedQuery, setSubmittedQuery] = useState("");
  const [mode, setMode] = useState<SearchMode>("hybrid");
  const [extension, setExtension] = useState("");
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [selected, setSelected] = useState<SearchResult | null>(null);
  const [searching, setSearching] = useState(false);
  const [inventory, setInventory] = useState<InventoryState | null>(null);
  const [inventoryLoading, setInventoryLoading] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const searchRequestId = useRef(0);

  const refreshStats = useCallback(async () => {
    try {
      const next = await coreApi.stats();
      setStats(next);
      setConnected(true);
    } catch {
      setConnected(false);
    }
  }, []);

  useEffect(() => {
    const start = async () => {
      try {
        if (isTauri()) await invoke("ensure_search_core");
        await refreshStats();
      } catch (error) {
        setNotice(`${t.error}: ${String(error)}`);
      }
    };
    void start();
    const timer = window.setInterval(refreshStats, 1200);
    return () => window.clearInterval(timer);
  }, [refreshStats, t.error]);

  const indexFolders = useCallback(
    async (paths: string[]) => {
      try {
        await coreApi.index(paths);
        setNotice(t.indexStarted);
        await refreshStats();
      } catch (error) {
        setNotice(`${t.error}: ${error instanceof Error ? error.message : String(error)}`);
      }
    },
    [refreshStats, t.error, t.indexStarted],
  );

  const addFolder = async () => {
    if (!isTauri()) {
      setNotice("请在 Tauri 桌面应用中选择目录");
      return;
    }
    const selectedPath = await open({ directory: true, multiple: false, title: t.addFolder });
    if (typeof selectedPath === "string" && !stats.directories.includes(selectedPath)) {
      await indexFolders([...stats.directories, selectedPath]);
    }
  };

  const runSearch = useCallback(async (searchQuery: string, searchMode: SearchMode, searchExtension: string, updateSubmittedQuery: boolean) => {
    const requestId = ++searchRequestId.current;
    setSearching(true);
    try {
      const result = await coreApi.search(searchQuery, searchMode, searchExtension);
      if (requestId !== searchRequestId.current) return;
      setResponse(result);
      if (updateSubmittedQuery) setSubmittedQuery(searchQuery);
      setSelected(result.results[0] ?? null);
    } catch (error) {
      if (requestId === searchRequestId.current) {
        setNotice(`${t.error}: ${error instanceof Error ? error.message : String(error)}`);
      }
    } finally {
      if (requestId === searchRequestId.current) setSearching(false);
    }
  }, [t.error]);

  const submitSearch = async (event: FormEvent) => {
    event.preventDefault();
    const searchQuery = query.trim();
    if (!searchQuery) return;
    await runSearch(searchQuery, mode, extension, true);
  };

  useEffect(() => {
    if (!submittedQuery.trim()) return;
    void runSearch(submittedQuery, mode, extension, false);
  }, [mode, extension, runSearch]);

  const showInventory = async (kind: InventoryKind, offset = 0) => {
    setInventoryLoading(true);
    try {
      if (kind === "documents") {
        setInventory({ kind, page: await coreApi.documents(offset) });
      } else if (kind === "chunks") {
        setInventory({ kind, page: await coreApi.chunks(offset) });
      } else {
        setInventory({ kind, page: await coreApi.failures(offset) });
      }
    } catch (error) {
      setNotice(`${t.error}: ${String(error)}`);
    } finally {
      setInventoryLoading(false);
    }
  };

  const openPath = async (path: string, reveal = false) => {
    try {
      if (!isTauri()) throw new Error("仅桌面应用支持此操作");
      await invoke(reveal ? "reveal_path" : "open_path", { path });
    } catch (error) {
      setNotice(`${t.error}: ${String(error)}`);
    }
  };

  const extensions = stats.extensions;
  const progress = stats.total_files ? Math.round((stats.processed_files / stats.total_files) * 100) : 0;
  const modelLoading = stats.embedding_model?.includes("正在加载") ?? false;
  const operation = modelLoading
    ? t.embeddingLoading
    : stats.total_files === 0
      ? t.scanning
    : stats.current_file
      ? `${t.processingFile}: ${basename(stats.current_file)}`
      : t.writingIndex;
  const backend = stats.embedding_backend ?? "cpu";
  const backendLabel = backend === "cuda"
    ? t.nvidiaCuda
    : backend === "fallback"
      ? t.offlineEmbedding
      : t.cpu;
  const stageLabels: Record<string, string> = {
    starting: t.indexing,
    scanning: t.stageScanning,
    checking: t.stageChecking,
    parsing: t.stageParsing,
    embedding: t.stageEmbedding,
    storage: t.stageStorage,
    text_index: t.stageTextIndex,
    committing: t.stageCommitting,
    ready: t.ready,
  };
  const timingRows = [
    ["scanning", t.stageScanning, stats.index_scan_ms ?? 0],
    ["checking", t.stageChecking, stats.index_check_ms ?? 0],
    ["parsing", t.stageParsing, stats.index_parse_ms ?? 0],
    ["embedding", t.stageEmbedding, stats.index_embedding_ms ?? 0],
    ["storage", t.stageStorage, stats.index_storage_ms ?? 0],
    ["text_index", t.stageTextIndex, stats.index_text_index_ms ?? 0],
  ] as const;
  const showTiming = stats.status === "indexing" || (stats.index_total_elapsed_ms ?? 0) > 0;

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand"><FileSearch aria-hidden="true" /><strong>{t.appName}</strong></div>
        <form className="searchbox" onSubmit={submitSearch}>
          <Search aria-hidden="true" />
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder={t.searchPlaceholder} />
          {query && <button className="icon-button" type="button" title={t.close} onClick={() => setQuery("")}><X /></button>}
          <button className="search-button" type="submit" disabled={!connected || searching} title={t.results}>
            {searching ? <RefreshCw className="spin" /> : <ChevronRight />}
          </button>
        </form>
        <label className="locale-control" title="Language">
          <Globe2 aria-hidden="true" />
          <select value={locale} onChange={(event) => setLocale(event.target.value as Locale)}>
            <option value="zh">中文</option><option value="en">EN</option><option value="ru">RU</option>
          </select>
        </label>
      </header>

      <div className="workspace">
        <aside className="sidebar">
          <div className="sidebar-heading"><span>{t.folders}</span><button className="icon-button" onClick={addFolder} title={t.addFolder}><Plus /></button></div>
          <div className="folder-list">
            {stats.directories.length === 0 && <p className="muted compact">{t.noFolder}</p>}
            {stats.directories.map((path) => (
              <div className="folder-row" key={path}>
                <button className="folder-open" title={`${t.openFolder}: ${path}`} onClick={() => void openPath(path)}>
                  <Folder aria-hidden="true" />
                  <span>{basename(path)}</span>
                </button>
                <button className="icon-button subtle" title={t.remove} onClick={() => void indexFolders(stats.directories.filter((item) => item !== path))}><Trash2 /></button>
              </div>
            ))}
          </div>
          <button className="add-folder" onClick={addFolder}><Plus />{t.addFolder}</button>

          <div className="index-summary">
            <div className="status-line"><span className={`status-dot ${connected ? stats.status : "offline"}`} />{!connected ? t.unavailable : stats.status === "indexing" ? operation : t.ready}</div>
            <div className={`backend-status ${backend}`} title={stats.embedding_model ?? undefined}>
              <Cpu aria-hidden="true" />
              <span>{t.embeddingBackend}: <strong>{backendLabel}</strong></span>
            </div>
            {stats.status === "indexing" && (
              <div className="progress-block">
                <div className={`progress-track ${modelLoading ? "indeterminate" : ""}`}><span style={modelLoading ? undefined : { width: `${progress}%` }} /></div>
                <small>{modelLoading ? stats.embedding_model : `${stats.processed_files} / ${stats.total_files}`}</small>
              </div>
            )}
            {showTiming && (
              <section className="timing-panel">
                <header><span>{t.timingTitle}</span><strong>{formatDuration(stats.index_total_elapsed_ms ?? 0)}</strong></header>
                {stats.status === "indexing" && (
                  <div className="timing-current">
                    <span>{t.currentStage}</span>
                    <strong>{stageLabels[stats.index_stage ?? "starting"]}</strong>
                    <time>{formatDuration(stats.index_stage_elapsed_ms ?? 0)}</time>
                  </div>
                )}
                <div className="timing-list">
                  {timingRows.map(([stage, label, elapsed]) => (
                    <div className={stats.status === "indexing" && stats.index_stage === stage ? "active" : ""} key={stage}>
                      <span>{label}</span><time>{formatDuration(elapsed + (stats.status === "indexing" && stats.index_stage === stage ? stats.index_stage_elapsed_ms ?? 0 : 0))}</time>
                    </div>
                  ))}
                </div>
              </section>
            )}
            <div className="stat-grid">
              <button onClick={() => void showInventory("documents")}><strong>{stats.document_count}</strong>{t.documents}</button>
              <button onClick={() => void showInventory("chunks")}><strong>{stats.chunk_count}</strong>{t.chunks}</button>
              <button onClick={() => void showInventory("failures")}><strong>{stats.failed_count}</strong>{t.failures}</button>
            </div>
            <div className="index-actions">
              <span><Clock3 />{stats.last_indexed ? formatDate(stats.last_indexed, locale) : t.never}</span>
              <button className="icon-button" disabled={stats.status === "indexing" || stats.directories.length === 0} onClick={() => void indexFolders(stats.directories)} title={t.reindex}><RefreshCw /></button>
            </div>
          </div>
        </aside>

        <main className="content">
          <div className="result-toolbar">
            <div><h1>{t.results}</h1>{response && <span>{t.resultCount(response.total, response.elapsed_ms)}</span>}</div>
            <div className="filters">
              <select value={extension} onChange={(event) => setExtension(event.target.value)} aria-label={t.allTypes}>
                <option value="">{t.allTypes}</option>
                {extensions.map((item) => <option value={item} key={item}>.{item}</option>)}
              </select>
              <div className="segmented" role="group">
                {(["hybrid", "keyword", "semantic"] as SearchMode[]).map((item) => (
                  <button type="button" className={mode === item ? "active" : ""} onClick={() => setMode(item)} key={item}>{t[item]}</button>
                ))}
              </div>
            </div>
          </div>

          {!response ? (
            <EmptyState hasFolders={stats.directories.length > 0} title={stats.directories.length ? t.searchTitle : t.startTitle} action={stats.directories.length ? t.searchAction : t.startAction} onAdd={addFolder} />
          ) : response.results.length === 0 ? (
            <EmptyState hasFolders title={t.searchTitle} action={t.searchAction} onAdd={addFolder} />
          ) : (
            <div className={`result-layout ${selected ? "with-detail" : ""}`}>
              <section className="result-list">
                {response.results.map((result) => (
                  <button className={`result-card ${selected?.id === result.id ? "selected" : ""}`} key={result.id} onClick={() => setSelected(result)}>
                    <FileBadge extension={result.extension} />
                    <span className="result-body">
                      <span className="result-title"><strong>{result.name}</strong><em>{Math.round(result.score * 100)}%</em></span>
                      <span className="result-path">{result.path}</span>
                      <span className="snippet"><Highlight text={result.snippet} query={submittedQuery} /></span>
                      <span className="result-meta">{formatBytes(result.size)} · {formatTimestamp(result.modified_ms, locale)}</span>
                    </span>
                  </button>
                ))}
              </section>
              {selected && <DetailPanel result={selected} t={t} locale={locale} onClose={() => setSelected(null)} onOpen={openPath} />}
            </div>
          )}
        </main>
      </div>

      {inventory && <InventoryModal inventory={inventory} loading={inventoryLoading} t={t} locale={locale} onClose={() => setInventory(null)} onPage={(offset) => void showInventory(inventory.kind, offset)} onOpen={openPath} />}
      {notice && <button className="toast" onClick={() => setNotice(null)}>{notice}<X /></button>}
    </div>
  );
}

function InventoryModal({
  inventory,
  loading,
  t,
  locale,
  onClose,
  onPage,
  onOpen,
}: {
  inventory: InventoryState;
  loading: boolean;
  t: ReturnType<typeof getMessages>;
  locale: Locale;
  onClose: () => void;
  onPage: (offset: number) => void;
  onOpen: (path: string) => void;
}) {
  const { page } = inventory;
  const title = inventory.kind === "documents" ? t.indexedDocuments : inventory.kind === "chunks" ? t.indexedChunks : t.failedFiles;
  const emptyText = inventory.kind === "documents" ? t.noDocuments : inventory.kind === "chunks" ? t.noChunks : t.noFailures;
  const start = page.total === 0 ? 0 : page.offset + 1;
  const end = Math.min(page.offset + page.items.length, page.total);

  return (
    <div className="modal-backdrop" onMouseDown={onClose}>
      <section className="modal inventory-modal" onMouseDown={(event) => event.stopPropagation()}>
        <header>
          <div>{inventory.kind === "failures" ? <AlertTriangle /> : <FileSearch />}<h2>{title}</h2></div>
          <button className="icon-button" onClick={onClose} title={t.close}><X /></button>
        </header>
        <div className={`inventory-list ${loading ? "loading" : ""}`}>
          {page.items.length === 0 && <p className="muted inventory-empty">{emptyText}</p>}
          {inventory.kind === "documents" && inventory.page.items.map((document) => (
            <button className="inventory-item" key={document.id} onClick={() => onOpen(document.path)}>
              <FileBadge extension={document.extension} />
              <span className="inventory-body">
                <strong>{document.name}</strong>
                <span className="inventory-path">{document.path}</span>
                <small>{formatBytes(document.size)} · {formatTimestamp(document.modified_ms, locale)}</small>
              </span>
              <ExternalLink />
            </button>
          ))}
          {inventory.kind === "chunks" && inventory.page.items.map((chunk) => (
            <button className="inventory-item chunk-item" key={chunk.id} onClick={() => onOpen(chunk.document_path)}>
              <File aria-hidden="true" />
              <span className="inventory-body">
                <strong>{chunk.document_name} · #{chunk.position + 1}</strong>
                <span className="inventory-path">{chunk.document_path}</span>
                <p>{chunk.text}</p>
              </span>
              <ExternalLink />
            </button>
          ))}
          {inventory.kind === "failures" && inventory.page.items.map((failure) => (
            <div className="inventory-item failure-item" key={failure.path}>
              <AlertTriangle />
              <span className="inventory-body">
                <strong>{failure.path}</strong>
                <small className="failure-category">{failure.category}</small>
                <p>{failure.reason}</p>
              </span>
            </div>
          ))}
        </div>
        <footer className="inventory-footer">
          <span>{t.listRange(start, end, page.total)}</span>
          <div>
            <button className="icon-button" disabled={loading || page.offset === 0} onClick={() => onPage(Math.max(0, page.offset - page.limit))} title={t.previous}><ChevronLeft /></button>
            {loading && <RefreshCw className="spin" />}
            <button className="icon-button" disabled={loading || page.offset + page.limit >= page.total} onClick={() => onPage(page.offset + page.limit)} title={t.next}><ChevronRight /></button>
          </div>
        </footer>
      </section>
    </div>
  );
}

function EmptyState({ hasFolders, title, action, onAdd }: { hasFolders: boolean; title: string; action: string; onAdd: () => void }) {
  return <div className="empty-state"><div className="empty-icon">{hasFolders ? <Search /> : <FolderOpen />}</div><h2>{title}</h2><p>{action}</p>{!hasFolders && <button className="primary-button" onClick={onAdd}><Plus />{action}</button>}</div>;
}

function DetailPanel({ result, t, locale, onClose, onOpen }: { result: SearchResult; t: ReturnType<typeof getMessages>; locale: Locale; onClose: () => void; onOpen: (path: string, reveal?: boolean) => void }) {
  return <aside className="detail-panel"><header><span>{t.details}</span><button className="icon-button" onClick={onClose} title={t.close}><X /></button></header><FileBadge extension={result.extension} large /><h2>{result.name}</h2><p className="detail-path">{result.path}</p><dl><div><dt>{t.modified}</dt><dd>{formatTimestamp(result.modified_ms, locale)}</dd></div><div><dt>{t.size}</dt><dd>{formatBytes(result.size)}</dd></div><div><dt>{t.relevance}</dt><dd>{Math.round(result.score * 100)}%</dd></div></dl><div className="detail-actions"><button className="primary-button" onClick={() => void onOpen(result.path)}><ExternalLink />{t.open}</button><button className="secondary-button" onClick={() => void onOpen(result.path, true)}><FolderOpen />{t.reveal}</button></div></aside>;
}

function FileBadge({ extension, large = false }: { extension: string; large?: boolean }) {
  return <span className={`file-badge type-${extension} ${large ? "large" : ""}`}><File aria-hidden="true" /><b>{extension.slice(0, 4).toUpperCase()}</b></span>;
}

function Highlight({ text, query }: { text: string; query: string }) {
  if (!query) return text;
  const index = text.toLowerCase().indexOf(query.toLowerCase());
  if (index < 0) return text;
  return <Fragment>{text.slice(0, index)}<mark>{text.slice(index, index + query.length)}</mark>{text.slice(index + query.length)}</Fragment>;
}

const basename = (path: string) => path.split(/[\\/]/).filter(Boolean).pop() ?? path;
const formatBytes = (size: number) => size < 1024 ? `${size} B` : size < 1024 ** 2 ? `${(size / 1024).toFixed(1)} KB` : `${(size / 1024 ** 2).toFixed(1)} MB`;
const formatDuration = (milliseconds: number) => milliseconds < 1_000
  ? `${Math.round(milliseconds)} ms`
  : milliseconds < 60_000
    ? `${(milliseconds / 1_000).toFixed(milliseconds < 10_000 ? 1 : 0)} s`
    : `${Math.floor(milliseconds / 60_000)}m ${Math.floor((milliseconds % 60_000) / 1_000)}s`;
const formatTimestamp = (value: number, locale: Locale) => new Intl.DateTimeFormat(locale === "zh" ? "zh-CN" : locale === "ru" ? "ru-RU" : "en-US", { dateStyle: "medium" }).format(new Date(value));
const formatDate = (value: string, locale: Locale) => formatTimestamp(new Date(value).getTime(), locale);

export default App;
