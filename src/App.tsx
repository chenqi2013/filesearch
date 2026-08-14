import { FormEvent, Fragment, useCallback, useEffect, useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import {
  AlertTriangle,
  ChevronRight,
  Clock3,
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
import type { IndexFailure, SearchMode, SearchResponse, SearchResult, ServiceStats } from "./types";

const EMPTY_STATS: ServiceStats = {
  status: "ready",
  document_count: 0,
  chunk_count: 0,
  failed_count: 0,
  processed_files: 0,
  total_files: 0,
  directories: [],
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
  const [failures, setFailures] = useState<IndexFailure[] | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

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

  const submitSearch = async (event: FormEvent) => {
    event.preventDefault();
    if (!query.trim()) return;
    setSearching(true);
    try {
      const result = await coreApi.search(query.trim(), mode, extension);
      setResponse(result);
      setSubmittedQuery(query.trim());
      setSelected(result.results[0] ?? null);
    } catch (error) {
      setNotice(`${t.error}: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setSearching(false);
    }
  };

  const showFailures = async () => {
    try {
      setFailures(await coreApi.failures());
    } catch (error) {
      setNotice(`${t.error}: ${String(error)}`);
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

  const extensions = useMemo(
    () => Array.from(new Set(response?.results.map((result) => result.extension) ?? [])),
    [response],
  );
  const progress = stats.total_files ? Math.round((stats.processed_files / stats.total_files) * 100) : 0;

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
                <Folder aria-hidden="true" />
                <span title={path}>{basename(path)}</span>
                <button className="icon-button subtle" title={t.remove} onClick={() => void indexFolders(stats.directories.filter((item) => item !== path))}><Trash2 /></button>
              </div>
            ))}
          </div>
          <button className="add-folder" onClick={addFolder}><Plus />{t.addFolder}</button>

          <div className="index-summary">
            <div className="status-line"><span className={`status-dot ${connected ? stats.status : "offline"}`} />{!connected ? t.unavailable : stats.status === "indexing" ? t.indexing : t.ready}</div>
            {stats.status === "indexing" && (
              <div className="progress-block"><div className="progress-track"><span style={{ width: `${progress}%` }} /></div><small>{stats.processed_files} / {stats.total_files}</small></div>
            )}
            <div className="stat-grid">
              <span><strong>{stats.document_count}</strong>{t.documents}</span>
              <span><strong>{stats.chunk_count}</strong>{t.chunks}</span>
              <button onClick={showFailures}><strong>{stats.failed_count}</strong>{t.failures}</button>
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
                  <button className={mode === item ? "active" : ""} onClick={() => setMode(item)} key={item}>{t[item]}</button>
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

      {failures && (
        <div className="modal-backdrop" onMouseDown={() => setFailures(null)}>
          <section className="modal" onMouseDown={(event) => event.stopPropagation()}>
            <header><div><AlertTriangle /><h2>{t.failedFiles}</h2></div><button className="icon-button" onClick={() => setFailures(null)} title={t.close}><X /></button></header>
            <div className="failure-list">
              {failures.length === 0 ? <p className="muted">{t.noFailures}</p> : failures.map((failure) => <div key={failure.path}><strong>{failure.path}</strong><p>{failure.reason}</p></div>)}
            </div>
          </section>
        </div>
      )}
      {notice && <button className="toast" onClick={() => setNotice(null)}>{notice}<X /></button>}
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
const formatTimestamp = (value: number, locale: Locale) => new Intl.DateTimeFormat(locale === "zh" ? "zh-CN" : locale === "ru" ? "ru-RU" : "en-US", { dateStyle: "medium" }).format(new Date(value));
const formatDate = (value: string, locale: Locale) => formatTimestamp(new Date(value).getTime(), locale);

export default App;

