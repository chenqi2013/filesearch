import { FormEvent, Fragment, PointerEvent as ReactPointerEvent, useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import {
  AlertTriangle,
  ChartNoAxesCombined,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  Clock3,
  Cpu,
  Download,
  ExternalLink,
  File,
  FileSearch,
  Folder,
  FolderOpen,
  Globe2,
  MoreHorizontal,
  Minimize2,
  PanelLeftClose,
  PanelLeftOpen,
  Pause,
  Play,
  Plus,
  RefreshCw,
  Search,
  Square,
  Trash2,
  X,
} from "lucide-react";
import { coreApi, CoreRequestError } from "./api";
import { getMessages, type Locale } from "./i18n";
import type {
  IndexFailure,
  IndexedChunk,
  IndexedDocument,
  FileIndexTiming,
  Page,
  SearchMode,
  SearchResponse,
  SearchResult,
  ServiceStats,
  ServiceDiagnostics,
} from "./types";

type InventoryKind = "documents" | "chunks" | "failures";
type ServiceIssue = { kind: string; message: string; time: number; path?: string; file?: string; stage?: string };
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
const DEFAULT_SIDEBAR_WIDTH = 258;
const MIN_SIDEBAR_WIDTH = 200;
const MAX_SIDEBAR_WIDTH = 420;

function App() {
  const [locale, setLocale] = useState<Locale>("zh");
  const t = getMessages(locale);
  const [stats, setStats] = useState<ServiceStats>(EMPTY_STATS);
  const [connected, setConnected] = useState(false);
  const [serviceDiagnostics, setServiceDiagnostics] = useState<ServiceDiagnostics | null>(null);
  const [serviceIssue, setServiceIssue] = useState<ServiceIssue | null>(null);
  const [serviceDetailsOpen, setServiceDetailsOpen] = useState(false);
  const [serviceRetrying, setServiceRetrying] = useState(false);
  const [query, setQuery] = useState("");
  const [submittedQuery, setSubmittedQuery] = useState("");
  const [mode, setMode] = useState<SearchMode>("hybrid");
  const [extension, setExtension] = useState("");
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [selected, setSelected] = useState<SearchResult | null>(null);
  const [searching, setSearching] = useState(false);
  const [inventory, setInventory] = useState<InventoryState | null>(null);
  const [inventoryLoading, setInventoryLoading] = useState(false);
  const [timingReport, setTimingReport] = useState<Page<FileIndexTiming> | null>(null);
  const [timingLoading, setTimingLoading] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [officeSetupRequired, setOfficeSetupRequired] = useState(false);
  const [sidebarHidden, setSidebarHidden] = useState(false);
  const [sidebarWidth, setSidebarWidth] = useState(DEFAULT_SIDEBAR_WIDTH);
  const [indexSummaryCollapsed, setIndexSummaryCollapsed] = useState(false);
  const [folderToRemove, setFolderToRemove] = useState<string | null>(null);
  const [folderMenuOpen, setFolderMenuOpen] = useState(false);
  const [folderSearchOpen, setFolderSearchOpen] = useState(false);
  const [folderSearch, setFolderSearch] = useState("");
  const [folderSort, setFolderSort] = useState<"default" | "modified-desc" | "modified-asc">("default");
  const [folderModifiedTimes, setFolderModifiedTimes] = useState<Record<string, number>>({});
  const folderMenuRef = useRef<HTMLDivElement>(null);
  const searchRequestId = useRef(0);
  const previousIndexStatus = useRef<ServiceStats["status"] | null>(null);
  const sidebarResize = useRef<{ startX: number; startWidth: number } | null>(null);
  const refreshingStats = useRef(false);
  const lastStats = useRef<ServiceStats>(EMPTY_STATS);
  const lastDiagnosticCheck = useRef(0);

  const inspectService = useCallback(async (context?: string) => {
    if (!isTauri()) return;
    try {
      const diagnostic = await invoke<ServiceDiagnostics>("search_core_diagnostics", { context: context ?? null });
      setServiceDiagnostics(diagnostic);
      lastDiagnosticCheck.current = Date.now();
    } catch (error) {
      setServiceDiagnostics((current) => current ? { ...current, detail: `${current.detail}\n${String(error)}` } : null);
    }
  }, []);

  const recordServiceFailure = useCallback(async (error: unknown) => {
    const issue = {
      kind: error instanceof CoreRequestError ? error.kind : "network",
      message: error instanceof Error ? error.message : String(error),
      time: Date.now(),
      path: error instanceof CoreRequestError ? error.path : undefined,
      file: lastStats.current.current_file,
      stage: lastStats.current.index_stage,
    };
    setServiceIssue(issue);
    await inspectService(JSON.stringify(issue));
  }, [inspectService]);

  const beginSidebarResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    sidebarResize.current = { startX: event.clientX, startWidth: sidebarWidth };
    event.currentTarget.setPointerCapture(event.pointerId);
    document.body.classList.add("resizing-sidebar");
  };

  const continueSidebarResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!sidebarResize.current) return;
    const width = sidebarResize.current.startWidth + event.clientX - sidebarResize.current.startX;
    setSidebarWidth(Math.min(MAX_SIDEBAR_WIDTH, Math.max(MIN_SIDEBAR_WIDTH, width)));
  };

  const finishSidebarResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    sidebarResize.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    document.body.classList.remove("resizing-sidebar");
  };

  const refreshStats = useCallback(async () => {
    if (refreshingStats.current) return;
    refreshingStats.current = true;
    try {
      const next = await coreApi.stats();
      lastStats.current = next;
      const previous = previousIndexStatus.current;
      previousIndexStatus.current = next.status;
      setStats(next);
      setConnected(true);
      if (Date.now() - lastDiagnosticCheck.current > 15_000) await inspectService();
      if (
        (previous === "indexing" || previous === "paused")
        && next.status === "ready"
        && next.failed_count > 0
      ) {
        try {
          const failures = await coreApi.failures(0, 100);
          if (failures.items.some(isOfficeDependencyFailure)) {
            setOfficeSetupRequired(true);
          }
        } catch {}
      }
    } catch (error) {
      setConnected(false);
      await recordServiceFailure(error);
    } finally {
      refreshingStats.current = false;
    }
  }, [inspectService, recordServiceFailure]);

  useEffect(() => {
    const start = async () => {
      try {
        if (isTauri()) await invoke("ensure_search_core");
        await refreshStats();
      } catch (error) {
        setNotice(`${t.error}: ${String(error)}`);
        await recordServiceFailure(error);
      }
    };
    void start();
    const timer = window.setInterval(refreshStats, 1200);
    return () => window.clearInterval(timer);
  }, [refreshStats, recordServiceFailure, t.error]);

  const retryService = async () => {
    if (serviceRetrying) return;
    setServiceRetrying(true);
    try {
      if (isTauri()) await invoke("ensure_search_core", { retry: true });
      await refreshStats();
      await inspectService();
    } catch (error) {
      await recordServiceFailure(error);
    } finally {
      setServiceRetrying(false);
    }
  };

  const openServiceLogs = async () => {
    try { await invoke("open_service_logs"); }
    catch (error) { setNotice(`${t.error}: ${String(error)}`); }
  };

  const copyServiceDetails = async () => {
    try {
      await navigator.clipboard.writeText(JSON.stringify({ request: serviceIssue, diagnostics: serviceDiagnostics }, null, 2));
      setNotice(t.serviceCopied);
    } catch (error) { setNotice(`${t.error}: ${String(error)}`); }
  };

  const hideToBackground = async () => {
    try { await invoke("hide_to_tray"); }
    catch (error) { setNotice(`${t.error}: ${String(error)}`); }
  };

  useEffect(() => {
    if (notice !== t.indexStarted) return;

    const timer = window.setTimeout(() => {
      setNotice((current) => current === t.indexStarted ? null : current);
    }, 3_000);

    return () => window.clearTimeout(timer);
  }, [notice, t.indexStarted]);

  useEffect(() => {
    if (!isTauri() || stats.directories.length === 0) {
      setFolderModifiedTimes({});
      return;
    }
    void invoke<Record<string, number>>("directory_modified_times", { paths: stats.directories })
      .then(setFolderModifiedTimes)
      .catch(() => setFolderModifiedTimes({}));
  }, [stats.directories]);

  useEffect(() => {
    if (!folderMenuOpen) return;
    const closeMenu = (event: MouseEvent) => {
      if (!folderMenuRef.current?.contains(event.target as Node)) setFolderMenuOpen(false);
    };
    window.addEventListener("mousedown", closeMenu);
    return () => window.removeEventListener("mousedown", closeMenu);
  }, [folderMenuOpen]);

  const indexFolders = useCallback(
    async (paths: string[]) => {
      try {
        await coreApi.index(paths);
        setNotice(t.indexStarted);
        await refreshStats();
      } catch (error) {
        if (error instanceof CoreRequestError && error.kind !== "http") await recordServiceFailure(error);
        setNotice(`${t.error}: ${error instanceof Error ? error.message : String(error)}`);
      }
    },
    [refreshStats, recordServiceFailure, t.error, t.indexStarted],
  );

  const controlIndex = async (action: "pause" | "resume" | "cancel") => {
    try {
      const result = action === "pause"
        ? await coreApi.pauseIndex()
        : action === "resume"
          ? await coreApi.resumeIndex()
          : await coreApi.cancelIndex();
      if (!result.accepted) {
        setNotice(result.message);
        return;
      }
      setNotice(action === "pause" ? t.indexPaused : action === "cancel" ? t.indexCancelRequested : t.indexStarted);
      await refreshStats();
    } catch (error) {
      if (error instanceof CoreRequestError && error.kind !== "http") await recordServiceFailure(error);
      setNotice(`${t.error}: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

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
        if (error instanceof CoreRequestError && error.kind !== "http") await recordServiceFailure(error);
        setNotice(`${t.error}: ${error instanceof Error ? error.message : String(error)}`);
      }
    } finally {
      if (requestId === searchRequestId.current) setSearching(false);
    }
  }, [recordServiceFailure, t.error]);

  const submitSearch = async (event: FormEvent) => {
    event.preventDefault();
    const searchQuery = query.trim();
    if (!searchQuery) return;
    await runSearch(searchQuery, mode, extension, true);
  };

  const removeFolder = async () => {
    if (!folderToRemove) return;
    const remainingDirectories = stats.directories.filter((item) => item !== folderToRemove);
    setFolderToRemove(null);
    await indexFolders(remainingDirectories);
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
      if (error instanceof CoreRequestError && error.kind !== "http") await recordServiceFailure(error);
      setNotice(`${t.error}: ${String(error)}`);
    } finally {
      setInventoryLoading(false);
    }
  };

  const showTimingReport = async (offset = 0) => {
    setTimingLoading(true);
    try {
      setTimingReport(await coreApi.fileTimings(offset));
    } catch (error) {
      if (error instanceof CoreRequestError && error.kind !== "http") await recordServiceFailure(error);
      setNotice(`${t.error}: ${String(error)}`);
    } finally {
      setTimingLoading(false);
    }
  };

  const exportTimingReport = async () => {
    try {
      const first = await coreApi.fileTimings(0);
      const items = [...first.items];
      for (let offset = first.limit; offset < first.total; offset += first.limit) {
        items.push(...(await coreApi.fileTimings(offset, first.limit)).items);
      }
      const quote = (value: string | number) => `"${String(value).replace(/"/g, '""')}"`;
      const rows = [
        ["path", "extension", "size_bytes", "status", "check_ms", "read_parse_ms", "pdf_ocr_ms", "embedding_ms", "sqlite_ms", "text_index_ms", "total_ms", "recorded_at"],
        ...items.map((item) => [item.path, item.extension, item.size, item.status, item.check_ms, item.parse_ms, item.ocr_ms, item.embedding_ms, item.storage_ms, item.text_index_ms, item.total_ms, item.recorded_at]),
      ];
      const csv = `\uFEFF${rows.map((row) => row.map(quote).join(",")).join("\r\n")}`;
      const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `filesearch-index-timings-${new Date().toISOString().slice(0, 10)}.csv`;
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      if (error instanceof CoreRequestError && error.kind !== "http") await recordServiceFailure(error);
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

  const downloadLibreOffice = async () => {
    try {
      if (!isTauri()) throw new Error("仅桌面应用支持此操作");
      await invoke("open_libreoffice_download");
      setOfficeSetupRequired(false);
    } catch (error) {
      setNotice(`${t.error}: ${String(error)}`);
    }
  };

  const extensions = stats.extensions;
  const isIndexing = stats.status === "indexing" || stats.status === "paused";
  const indexDetailsCollapsed = indexSummaryCollapsed && !isIndexing;
  const progress = stats.total_files ? Math.round((stats.processed_files / stats.total_files) * 100) : 0;
  const modelLoading = stats.embedding_model?.includes("正在加载") ?? false;
  const operation = stats.status === "paused"
    ? t.indexPaused
    : modelLoading
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
  const showTiming = isIndexing || (stats.index_total_elapsed_ms ?? 0) > 0;
  const normalizedFolderSearch = folderSearch.trim().toLocaleLowerCase();
  const visibleDirectories = stats.directories
    .filter((path) => !normalizedFolderSearch || path.toLocaleLowerCase().includes(normalizedFolderSearch))
    .sort((left, right) => {
      if (folderSort === "default") return stats.directories.indexOf(left) - stats.directories.indexOf(right);
      const difference = (folderModifiedTimes[left] ?? 0) - (folderModifiedTimes[right] ?? 0);
      return folderSort === "modified-desc" ? -difference : difference;
    });

  const toggleFolderSort = () => {
    setFolderSort((current) => current === "modified-desc" ? "modified-asc" : "modified-desc");
    setFolderMenuOpen(false);
  };

  return (
    <div className="app-shell">
      <header className={`topbar ${sidebarHidden ? "sidebar-hidden" : ""}`}>
        <div className="brand">
          <FileSearch aria-hidden="true" />
          <strong>{t.appName}</strong>
          <button
            className="icon-button sidebar-toggle"
            type="button"
            title={sidebarHidden ? t.showSidebar : t.hideSidebar}
            aria-label={sidebarHidden ? t.showSidebar : t.hideSidebar}
            aria-expanded={!sidebarHidden}
            onClick={() => setSidebarHidden((hidden) => !hidden)}
          >
            {sidebarHidden ? <PanelLeftOpen /> : <PanelLeftClose />}
          </button>
          {isTauri() && <button className="icon-button" title={t.background} aria-label={t.background} onClick={() => void hideToBackground()}><Minimize2 /></button>}
        </div>
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

      <div
        className={`workspace ${sidebarHidden ? "sidebar-hidden" : ""}`}
        style={sidebarHidden ? undefined : { gridTemplateColumns: `${sidebarWidth}px 5px minmax(0, 1fr)` }}
      >
        {!sidebarHidden && <aside className="sidebar">
          <div className="sidebar-heading">
            <span>{t.folders}</span>
            {stats.directories.length > 0 && (
              <div className="folder-menu-wrap" ref={folderMenuRef}>
                <button className="icon-button" onClick={() => setFolderMenuOpen((open) => !open)} title={t.more} aria-expanded={folderMenuOpen}><MoreHorizontal /></button>
                {folderMenuOpen && (
                  <div className="folder-menu">
                    <button onClick={() => { setFolderMenuOpen(false); void addFolder(); }}><Plus />{t.addFolder}</button>
                    <button onClick={() => { setFolderSearchOpen(true); setFolderMenuOpen(false); }}><Search />{t.searchFolders}</button>
                    <button onClick={toggleFolderSort}><Clock3 />{folderSort === "modified-desc" ? t.sortFoldersOldest : t.sortFoldersNewest}</button>
                  </div>
                )}
              </div>
            )}
          </div>
          {folderSearchOpen && stats.directories.length > 0 && (
            <div className="folder-search">
              <Search />
              <input autoFocus value={folderSearch} onChange={(event) => setFolderSearch(event.target.value)} placeholder={t.searchFoldersPlaceholder} />
              <button className="icon-button" title={t.close} onClick={() => { setFolderSearch(""); setFolderSearchOpen(false); }}><X /></button>
            </div>
          )}
          <div className="folder-list">
            {stats.directories.length === 0 && <p className="muted compact">{t.noFolder}</p>}
            {stats.directories.length > 0 && visibleDirectories.length === 0 && <p className="muted compact">{t.noMatchingFolders}</p>}
            {visibleDirectories.map((path) => (
              <div className="folder-row" key={path}>
                <button className="folder-open" title={`${t.openFolder}: ${path}`} onClick={() => void openPath(path)}>
                  <Folder aria-hidden="true" />
                  <span>{basename(path)}</span>
                </button>
                <button className="icon-button subtle" title={t.remove} onClick={() => setFolderToRemove(path)}><Trash2 /></button>
              </div>
            ))}
          </div>
          {stats.directories.length === 0 && <button className="add-folder" onClick={addFolder}><Plus />{t.addFolder}</button>}

          <div className="index-summary">
            <div className="status-line">
              <span className={`status-dot ${connected ? stats.status : "offline"}`} />
              <span
                className="status-label"
                title={stats.current_file ? basename(stats.current_file) : undefined}
              >
                {!connected ? t.unavailable : isIndexing ? operation : t.ready}
              </span>
              {connected && !isIndexing && (
                <button
                  className="icon-button summary-toggle"
                  onClick={() => setIndexSummaryCollapsed((collapsed) => !collapsed)}
                  title={indexSummaryCollapsed ? t.expandIndexInfo : t.collapseIndexInfo}
                  aria-label={indexSummaryCollapsed ? t.expandIndexInfo : t.collapseIndexInfo}
                  aria-expanded={!indexSummaryCollapsed}
                >
                  {indexSummaryCollapsed ? <ChevronDown /> : <ChevronUp />}
                </button>
              )}
            </div>
            {(!connected || serviceIssue || serviceDiagnostics?.last_exit) && (
              <div className={`service-alert ${connected ? "recovered" : "disconnected"}`} role="status">
                <span>{connected ? t.serviceRecovered : serviceIssue?.kind === "timeout" ? t.serviceTimeout : serviceIssue?.kind === "invalid_response" ? t.serviceResponse : serviceDiagnostics?.detail ?? t.serviceNetwork}</span>
                <button onClick={() => { setServiceDetailsOpen(true); void inspectService(); }}>{t.serviceDetails}</button>
                {!connected && <button disabled={serviceRetrying} onClick={() => void retryService()}>{t.serviceReconnect}</button>}
              </div>
            )}
            {!indexDetailsCollapsed && (
              <>
                <div className={`backend-status ${backend}`} title={stats.embedding_model ?? undefined}>
                  <Cpu aria-hidden="true" />
                  <span>{t.embeddingBackend}: <strong>{backendLabel}</strong></span>
                </div>
                {isIndexing && (
                  <div className="progress-block">
                    <div className={`progress-track ${modelLoading ? "indeterminate" : ""}`}><span style={modelLoading ? undefined : { width: `${progress}%` }} /></div>
                    <small>{modelLoading ? stats.embedding_model : `${stats.processed_files} / ${stats.total_files}`}</small>
                    {(stats.embedding_windows_total ?? 0) > 0 && (
                      <small>{t.embeddingWindows}: {stats.embedding_windows_processed ?? 0} / {stats.embedding_windows_total}</small>
                    )}
                  </div>
                )}
                {showTiming && (
                  <section className="timing-panel">
                    <header><span>{t.timingTitle}</span><strong>{formatDuration(stats.index_total_elapsed_ms ?? 0)}</strong></header>
                    {isIndexing && (
                      <div className="timing-current">
                        <span>{t.currentStage}</span>
                        <strong>{stageLabels[stats.index_stage ?? "starting"]}</strong>
                        <time>{formatDuration(stats.index_stage_elapsed_ms ?? 0)}</time>
                      </div>
                    )}
                    <div className="timing-list">
                      {timingRows.map(([stage, label, elapsed]) => (
                        <div className={isIndexing && stats.index_stage === stage ? "active" : ""} key={stage}>
                          <span>{label}</span><time>{formatDuration(elapsed + (isIndexing && stats.index_stage === stage ? stats.index_stage_elapsed_ms ?? 0 : 0))}</time>
                        </div>
                      ))}
                    </div>
                  </section>
                )}
              </>
            )}
            <div className="stat-grid">
              <button onClick={() => void showInventory("documents")}><strong>{stats.document_count}</strong>{t.documents}</button>
              <button onClick={() => void showInventory("chunks")}><strong>{stats.chunk_count}</strong>{t.chunks}</button>
              <button onClick={() => void showInventory("failures")}><strong>{stats.failed_count}</strong>{t.failures}</button>
            </div>
            {!indexDetailsCollapsed && <div className="index-actions">
              <span><Clock3 />{stats.last_indexed ? formatDate(stats.last_indexed, locale) : t.never}</span>
              <div className="index-action-buttons">
                <button className="icon-button" onClick={() => void showTimingReport()} title={t.fileTimingReport} aria-label={t.fileTimingReport}><ChartNoAxesCombined /></button>
                {stats.status === "indexing" && <button className="icon-button" onClick={() => void controlIndex("pause")} title={t.pauseIndex}><Pause /></button>}
                {stats.status === "paused" && <button className="icon-button" onClick={() => void controlIndex("resume")} title={t.resumeIndex}><Play /></button>}
                {isIndexing && <button className="icon-button danger-icon" onClick={() => void controlIndex("cancel")} title={t.cancelIndex}><Square /></button>}
                {!isIndexing && <button className="icon-button" disabled={stats.directories.length === 0} onClick={() => void indexFolders(stats.directories)} title={t.reindex}><RefreshCw /></button>}
              </div>
            </div>}
          </div>
        </aside>}
        {!sidebarHidden && (
          <div
            className="sidebar-resizer"
            role="separator"
            aria-orientation="vertical"
            aria-label={t.resizeSidebar}
            aria-valuemin={MIN_SIDEBAR_WIDTH}
            aria-valuemax={MAX_SIDEBAR_WIDTH}
            aria-valuenow={Math.round(sidebarWidth)}
            title={t.resizeSidebar}
            onPointerDown={beginSidebarResize}
            onPointerMove={continueSidebarResize}
            onPointerUp={finishSidebarResize}
            onPointerCancel={finishSidebarResize}
            onDoubleClick={() => setSidebarWidth(DEFAULT_SIDEBAR_WIDTH)}
          />
        )}

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
                      <span className="result-title"><strong><Highlight text={result.name} query={submittedQuery} /></strong><em>{Math.round(result.score * 100)}%</em></span>
                      <span className="result-path">{result.path}</span>
                      <span className="snippet"><Highlight text={result.snippet} query={submittedQuery} /></span>
                      <span className="result-meta">{formatBytes(result.size)} · {formatTimestamp(result.modified_ms, locale)}{Boolean(result.duplicates?.length) && <span className="duplicate-count">{t.duplicateCount(result.duplicates!.length)}</span>}</span>
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
      {timingReport && <FileTimingModal page={timingReport} loading={timingLoading} t={t} onClose={() => setTimingReport(null)} onPage={(offset) => void showTimingReport(offset)} onOpen={openPath} onExport={() => void exportTimingReport()} />}
      {serviceDetailsOpen && (
        <div className="modal-backdrop" onMouseDown={() => setServiceDetailsOpen(false)}>
          <section className="modal service-modal" role="dialog" aria-modal="true" aria-labelledby="service-title" onMouseDown={(event) => event.stopPropagation()}>
            <header><h2 id="service-title">{t.serviceDetails}</h2><button className="icon-button" title={t.close} onClick={() => setServiceDetailsOpen(false)}><X /></button></header>
            <div className="service-details">
              <p>{serviceDiagnostics?.detail ?? (serviceIssue?.kind === "timeout" ? t.serviceTimeout : t.serviceNetwork)}</p>
              <dl>
                <dt>{t.serviceBuild}</dt><dd>{serviceDiagnostics?.build ?? "—"}</dd>
                <dt>{t.serviceProcess}</dt><dd>{serviceDiagnostics?.pid ?? "—"}</dd>
                <dt>{t.serviceLastExit}</dt><dd>{serviceDiagnostics?.last_exit ? `${formatTimestamp(serviceDiagnostics.last_exit.timestamp_ms, locale)} · ${serviceDiagnostics.last_exit.reason}` : t.serviceUnknown}</dd>
                <dt>{t.serviceRequest}</dt><dd>{serviceIssue ? `${formatTimestamp(serviceIssue.time, locale)} · ${serviceIssue.path ?? ""}\n${serviceIssue.message}` : "—"}</dd>
                <dt>{t.serviceLastFile}</dt><dd>{serviceIssue?.file ?? "—"}</dd>
                <dt>{t.serviceLastStage}</dt><dd>{serviceIssue?.stage ?? "—"}</dd>
                <dt>{t.serviceLogs}</dt><dd>{serviceDiagnostics?.log_path ?? "—"}</dd>
              </dl>
              {serviceDiagnostics?.log_tail && <pre>{serviceDiagnostics.log_tail}</pre>}
              <p className="muted">{t.servicePrivacy}</p>
              <p className="muted">{t.backgroundHint}</p>
            </div>
            <footer>
              <button className="secondary-button" onClick={() => void copyServiceDetails()}>{t.serviceCopy}</button>
              {isTauri() && <button className="secondary-button" onClick={() => void openServiceLogs()}>{t.serviceLogs}</button>}
              <button className="primary-button" disabled={serviceRetrying} onClick={() => void retryService()}>{t.serviceReconnect}</button>
            </footer>
          </section>
        </div>
      )}
      {folderToRemove && (
        <ConfirmRemoveFolderModal
          path={folderToRemove}
          t={t}
          onCancel={() => setFolderToRemove(null)}
          onConfirm={() => void removeFolder()}
        />
      )}
      {officeSetupRequired && (
        <OfficeSetupModal
          t={t}
          onCancel={() => setOfficeSetupRequired(false)}
          onDownload={() => void downloadLibreOffice()}
        />
      )}
      {notice && <button className="toast" onClick={() => setNotice(null)}>{notice}<X /></button>}
    </div>
  );
}

function OfficeSetupModal({ t, onCancel, onDownload }: {
  t: ReturnType<typeof getMessages>;
  onCancel: () => void;
  onDownload: () => void;
}) {
  return (
    <div className="modal-backdrop" onMouseDown={onCancel}>
      <section className="modal confirm-modal" role="alertdialog" aria-modal="true" aria-labelledby="office-required-title" onMouseDown={(event) => event.stopPropagation()}>
        <header>
          <div><AlertTriangle /><h2 id="office-required-title">{t.officeRequiredTitle}</h2></div>
          <button className="icon-button" onClick={onCancel} title={t.close}><X /></button>
        </header>
        <div className="confirm-modal-body">
          <p>{t.officeRequiredMessage}</p>
          <small>{t.officeRequiredHint}</small>
        </div>
        <footer className="confirm-modal-actions">
          <button className="secondary-button" onClick={onCancel}>{t.cancel}</button>
          <button className="primary-button" onClick={onDownload}><ExternalLink />{t.downloadLibreOffice}</button>
        </footer>
      </section>
    </div>
  );
}

function ConfirmRemoveFolderModal({ path, t, onCancel, onConfirm }: {
  path: string;
  t: ReturnType<typeof getMessages>;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  return (
    <div className="modal-backdrop" onMouseDown={onCancel}>
      <section className="modal confirm-modal" role="alertdialog" aria-modal="true" aria-labelledby="remove-folder-title" onMouseDown={(event) => event.stopPropagation()}>
        <header>
          <div><AlertTriangle /><h2 id="remove-folder-title">{t.removeConfirmTitle}</h2></div>
          <button className="icon-button" onClick={onCancel} title={t.close}><X /></button>
        </header>
        <div className="confirm-modal-body">
          <p>{t.removeConfirmMessage}</p>
          <strong title={path}>{path}</strong>
          <small>{t.removeConfirmHint}</small>
        </div>
        <footer className="confirm-modal-actions">
          <button className="secondary-button" onClick={onCancel}>{t.cancel}</button>
          <button className="danger-button" onClick={onConfirm}><Trash2 />{t.removeConfirmAction}</button>
        </footer>
      </section>
    </div>
  );

}

function FileTimingModal({ page, loading, t, onClose, onPage, onOpen, onExport }: {
  page: Page<FileIndexTiming>;
  loading: boolean;
  t: ReturnType<typeof getMessages>;
  onClose: () => void;
  onPage: (offset: number) => void;
  onOpen: (path: string) => void;
  onExport: () => void;
}) {
  const totalPages = Math.max(1, Math.ceil(page.total / page.limit));
  const currentPage = Math.min(totalPages, Math.floor(page.offset / page.limit) + 1);
  const statusLabel = (status: FileIndexTiming["status"]) => status === "failed" ? t.timingFailed : status === "skipped" ? t.timingSkipped : t.timingIndexed;
  return (
    <div className="modal-backdrop" onMouseDown={onClose}>
      <section className="modal timing-modal" onMouseDown={(event) => event.stopPropagation()}>
        <header>
          <div><ChartNoAxesCombined /><h2>{t.fileTimingReport}</h2></div>
          <div className="timing-modal-actions">
            <button className="secondary-button" onClick={onExport} disabled={page.total === 0}><Download />{t.timingExport}</button>
            <button className="icon-button" onClick={onClose} title={t.close}><X /></button>
          </div>
        </header>
        <p className="timing-note">{t.timingNote}</p>
        <div className={`timing-table-wrap ${loading ? "loading" : ""}`}>
          {page.items.length === 0 ? <p className="muted inventory-empty">{t.timingNoData}</p> : (
            <table className="timing-table">
              <thead><tr><th>{t.timingStatus}</th><th>{t.size}</th><th>{t.stageChecking}</th><th>{t.timingParse}</th><th>{t.timingOcr}</th><th>{t.stageEmbedding}</th><th>{t.stageStorage}</th><th>{t.stageTextIndex}</th><th>{t.timingTitle}</th><th>{t.details}</th></tr></thead>
              <tbody>{page.items.map((item) => (
                <tr key={`${item.path}-${item.recorded_at}`}>
                  <td><span className={`timing-status ${item.status}`}>{statusLabel(item.status)}</span></td>
                  <td>{formatBytes(item.size)}</td>
                  <td>{formatDuration(item.check_ms)}</td>
                  <td>{formatDuration(item.parse_ms)}</td>
                  <td>{formatDuration(item.ocr_ms)}</td>
                  <td>{formatDuration(item.embedding_ms)}</td>
                  <td>{formatDuration(item.storage_ms)}</td>
                  <td>{formatDuration(item.text_index_ms)}</td>
                  <td><strong>{formatDuration(item.total_ms)}</strong></td>
                  <td><button className="timing-path" title={item.path} onClick={() => onOpen(item.path)}>{item.path}</button></td>
                </tr>
              ))}</tbody>
            </table>
          )}
        </div>
        <footer className="inventory-footer">
          <span>{t.listRange(page.total === 0 ? 0 : page.offset + 1, Math.min(page.offset + page.items.length, page.total), page.total)}</span>
          <div>
            <button className="icon-button" disabled={loading || page.offset === 0} onClick={() => onPage(Math.max(0, page.offset - page.limit))} title={t.previous}><ChevronLeft /></button>
            <select className="inventory-page-select" value={currentPage} disabled={loading || page.total === 0} aria-label={t.pageNumber(currentPage, totalPages)} onChange={(event) => onPage((Number(event.target.value) - 1) * page.limit)}>
              {Array.from({ length: totalPages }, (_, index) => <option key={index + 1} value={index + 1}>{t.pageNumber(index + 1, totalPages)}</option>)}
            </select>
            <button className="icon-button" disabled={loading || page.offset + page.limit >= page.total} onClick={() => onPage(page.offset + page.limit)} title={t.next}><ChevronRight /></button>
          </div>
        </footer>
      </section>
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
  const totalPages = Math.max(1, Math.ceil(page.total / page.limit));
  const currentPage = Math.min(totalPages, Math.floor(page.offset / page.limit) + 1);

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
            <button
              type="button"
              className="inventory-item failure-item"
              key={failure.path}
              title={`${t.open}: ${failure.path}`}
              onClick={() => onOpen(failure.path)}
            >
              <AlertTriangle />
              <span className="inventory-body">
                <strong>{failure.path}</strong>
                <small className="failure-category">{failure.category}</small>
                <p>{failure.reason}</p>
              </span>
              <ExternalLink />
            </button>
          ))}
        </div>
        <footer className="inventory-footer">
          <span>{t.listRange(start, end, page.total)}</span>
          <div>
            <button className="icon-button" disabled={loading || page.offset === 0} onClick={() => onPage(Math.max(0, page.offset - page.limit))} title={t.previous}><ChevronLeft /></button>
            {loading && <RefreshCw className="spin" />}
            <button className="icon-button" disabled={loading || page.offset + page.limit >= page.total} onClick={() => onPage(page.offset + page.limit)} title={t.next}><ChevronRight /></button>
            {inventory.kind !== "failures" && (
              <select
                className="inventory-page-select"
                value={currentPage}
                disabled={loading || page.total === 0}
                aria-label={t.pageNumber(currentPage, totalPages)}
                onChange={(event) => onPage((Number(event.target.value) - 1) * page.limit)}
              >
                {Array.from({ length: totalPages }, (_, index) => index + 1).map((pageNumber) => (
                  <option key={pageNumber} value={pageNumber}>{t.pageNumber(pageNumber, totalPages)}</option>
                ))}
              </select>
            )}
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
  return <aside className="detail-panel">
    <header><span>{t.details}</span><button className="icon-button" onClick={onClose} title={t.close}><X /></button></header>
    <FileBadge extension={result.extension} large />
    <h2>{result.name}</h2><p className="detail-path">{result.path}</p>
    <dl><div><dt>{t.modified}</dt><dd>{formatTimestamp(result.modified_ms, locale)}</dd></div><div><dt>{t.size}</dt><dd>{formatBytes(result.size)}</dd></div><div><dt>{t.relevance}</dt><dd>{Math.round(result.score * 100)}%</dd></div></dl>
    <div className="detail-actions">
      <button className="primary-button" onClick={() => void onOpen(result.path)}><ExternalLink />{t.open}</button>
      <button className="secondary-button" onClick={() => void onOpen(result.path, true)}><FolderOpen />{t.reveal}</button>
    </div>
    {Boolean(result.duplicates?.length) && <details className="duplicate-files" key={result.id}>
      <summary>{t.sameText} ({result.duplicates!.length + 1})</summary>
      {result.duplicates!.map((document) => <div className="duplicate-file" key={document.id}>
        <span><strong>{document.name}</strong><small>{document.path}</small></span>
        <button className="icon-button" title={t.open} aria-label={`${t.open}: ${document.path}`} onClick={() => void onOpen(document.path)}><ExternalLink /></button>
        <button className="icon-button" title={t.reveal} aria-label={`${t.reveal}: ${document.path}`} onClick={() => void onOpen(document.path, true)}><FolderOpen /></button>
      </div>)}
    </details>}
  </aside>;
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
const isOfficeDependencyFailure = (failure: IndexFailure) => failure.category === "dependency"
  || failure.reason.includes("旧版 Office 转换环境不可用");
const formatBytes = (size: number) => size < 1024 ? `${size} B` : size < 1024 ** 2 ? `${(size / 1024).toFixed(1)} KB` : `${(size / 1024 ** 2).toFixed(1)} MB`;
const formatDuration = (milliseconds: number) => milliseconds < 1_000
  ? `${Math.round(milliseconds)} ms`
  : milliseconds < 60_000
    ? `${(milliseconds / 1_000).toFixed(milliseconds < 10_000 ? 1 : 0)} s`
    : `${Math.floor(milliseconds / 60_000)}m ${Math.floor((milliseconds % 60_000) / 1_000)}s`;
const formatTimestamp = (value: number, locale: Locale) => new Intl.DateTimeFormat(locale === "zh" ? "zh-CN" : locale === "ru" ? "ru-RU" : "en-US", { dateStyle: "medium" }).format(new Date(value));
const formatDate = (value: string, locale: Locale) => formatTimestamp(new Date(value).getTime(), locale);

export default App;
