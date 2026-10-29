use serde::Serialize;
use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const RESTART_WINDOW: Duration = Duration::from_secs(60);
const MAX_RESTARTS: usize = 3;
const LOG_LIMIT: u64 = 10 * 1024 * 1024;

#[derive(Clone, Serialize)]
pub struct ServiceExit {
    pub timestamp_ms: u64,
    pub code: Option<i32>,
    pub reason: String,
}

#[derive(Clone, Serialize)]
pub struct ServiceDiagnostics {
    pub state: String,
    pub pid: Option<u32>,
    pub detail: String,
    pub last_exit: Option<ServiceExit>,
    pub log_path: String,
    pub log_tail: String,
    pub restart_attempts: usize,
    pub build: String,
}

pub struct ServiceConfig {
    pub executable: PathBuf,
    pub data_dir: PathBuf,
    pub model_dir: PathBuf,
    pub address: SocketAddr,
}

struct ServiceState {
    child: Option<Child>,
    stopped: bool,
    started: bool,
    last_exit: Option<ServiceExit>,
    attempts: VecDeque<Instant>,
    next_retry: Option<Instant>,
    failure: Option<String>,
    last_request: Option<(String, Instant)>,
}

pub struct ServiceSupervisor {
    config: ServiceConfig,
    state: Mutex<ServiceState>,
}

impl ServiceSupervisor {
    pub fn new(config: ServiceConfig) -> Self {
        Self {
            config,
            state: Mutex::new(ServiceState {
                child: None,
                stopped: false,
                started: false,
                last_exit: None,
                attempts: VecDeque::new(),
                next_retry: None,
                failure: None,
                last_request: None,
            }),
        }
    }

    pub fn monitor(supervisor: &Arc<Self>) {
        let weak = Arc::downgrade(supervisor);
        std::thread::spawn(move || loop {
            std::thread::sleep(Duration::from_secs(2));
            let Some(supervisor) = weak.upgrade() else {
                break;
            };
            let mut state = supervisor
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if state.stopped {
                break;
            }
            supervisor.observe_exit(&mut state);
            if state.started
                && state.child.is_none()
                && (state.last_exit.is_some() || state.failure.is_some())
            {
                let _ = supervisor.ensure_locked(&mut state);
            }
        });
    }

    pub fn ensure(&self, manual_retry: bool) -> Result<(), String> {
        let mut state = self.state.lock().map_err(|error| error.to_string())?;
        if state.stopped {
            return Err("应用正在退出".to_owned());
        }
        self.observe_exit(&mut state);
        state.started = true;
        if manual_retry && state.child.is_none() {
            state.attempts.clear();
            state.next_retry = None;
            state.failure = None;
        }
        self.ensure_locked(&mut state)
    }

    fn ensure_locked(&self, state: &mut ServiceState) -> Result<(), String> {
        if state.child.is_some() {
            return Ok(());
        }
        match probe(self.config.address) {
            Probe::Ready => {
                state.failure = None;
                return Ok(());
            }
            Probe::Unresponsive(detail) => {
                state.failure = Some(detail.clone());
                return Err(detail);
            }
            Probe::Unavailable(_) => {}
        }
        if state.next_retry.is_some_and(|retry| retry > Instant::now()) {
            return Err("搜索服务已退出，正在等待重试".to_owned());
        }
        if state.attempts.len() >= MAX_RESTARTS {
            return Err("搜索服务多次退出，已停止自动重试，请查看日志后手动重试".to_owned());
        }
        if state.last_exit.is_some() || state.failure.is_some() {
            state.attempts.push_back(Instant::now());
        }
        let result = self.spawn();
        match result {
            Ok(child) => {
                self.record(&format!("started pid={} build={}", child.id(), build_id()));
                state.child = Some(child);
                state.failure = None;
                state.next_retry = None;
                Ok(())
            }
            Err(error) => {
                self.record(&format!("start failed: {error}"));
                state.failure = Some(error.clone());
                state.next_retry = Some(Instant::now() + Duration::from_secs(5));
                Err(error)
            }
        }
    }

    fn spawn(&self) -> Result<Child, String> {
        std::fs::create_dir_all(self.log_dir())
            .map_err(|error| format!("无法创建日志目录: {error}"))?;
        rotate_log(&self.log_path()).map_err(|error| format!("无法轮转日志: {error}"))?;
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.log_path())
            .map_err(|error| format!("无法写入搜索服务日志: {error}"))?;
        let mut command = Command::new(&self.config.executable);
        command
            .args(["--listen", &self.config.address.to_string()])
            .arg("--data-dir")
            .arg(&self.config.data_dir)
            .arg("--model-dir")
            .arg(&self.config.model_dir)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env(
                "RUST_LOG",
                std::env::var("RUST_LOG")
                    .unwrap_or_else(|_| "info,tantivy=warn,tower_http=warn".to_owned()),
            )
            .env("RUST_BACKTRACE", "1");
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x0800_0000);
        }
        let mut child = command.spawn().map_err(|error| {
            format!(
                "无法启动搜索服务 {}: {error}",
                self.config.executable.display()
            )
        })?;
        let log_lock = Arc::new(Mutex::new(()));
        if let Some(output) = child.stdout.take() {
            capture_log(output, self.log_path(), Arc::clone(&log_lock));
        }
        if let Some(output) = child.stderr.take() {
            capture_log(output, self.log_path(), log_lock);
        }
        Ok(child)
    }

    fn observe_exit(&self, state: &mut ServiceState) {
        let status = match state.child.as_mut().map(Child::try_wait) {
            Some(Ok(Some(status))) => status,
            Some(Err(error)) => {
                state.failure = Some(format!("无法检查服务进程: {error}"));
                return;
            }
            _ => return,
        };
        let reason = exit_reason(status.code());
        self.record(&format!(
            "exited pid={:?} status={status} reason={reason}",
            state.child.as_ref().map(Child::id)
        ));
        state.child = None;
        state.last_exit = Some(ServiceExit {
            timestamp_ms: now_ms(),
            code: status.code(),
            reason,
        });
        let now = Instant::now();
        while state
            .attempts
            .front()
            .is_some_and(|time| now.duration_since(*time) > RESTART_WINDOW)
        {
            state.attempts.pop_front();
        }
        state.next_retry = Some(
            Instant::now()
                + Duration::from_secs(2_u64.pow(state.attempts.len() as u32 + 1).min(30)),
        );
    }

    pub fn diagnostics(&self, context: Option<&str>) -> ServiceDiagnostics {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        self.observe_exit(&mut state);
        if let Some(context) = context {
            let signature = request_signature(context);
            if state.last_request.as_ref().is_none_or(|(previous, time)| {
                previous != &signature || time.elapsed() > Duration::from_secs(15)
            }) {
                self.record(&format!(
                    "request failure: {}",
                    context.chars().take(2000).collect::<String>()
                ));
                state.last_request = Some((signature, Instant::now()));
            }
        }
        let (kind, detail) = if state.stopped {
            ("stopped", "搜索服务已停止".to_owned())
        } else {
            match probe(self.config.address) {
                Probe::Ready => {
                    state.failure = None;
                    ("running", "搜索服务健康接口正常".to_owned())
                }
                Probe::Unresponsive(detail) => ("unresponsive", detail),
                Probe::Unavailable(detail) if state.child.is_some() => (
                    "starting",
                    format!("服务进程仍在，接口暂不可连接；可能正在启动。{detail}"),
                ),
                Probe::Unavailable(_) if state.attempts.len() >= MAX_RESTARTS => (
                    "recovery_failed",
                    format!(
                        "搜索服务多次退出或启动失败，已停止自动重试，请查看日志后手动重试。{}",
                        state.failure.as_deref().unwrap_or_default()
                    ),
                ),
                Probe::Unavailable(_) if state.last_exit.is_some() || state.failure.is_some() => (
                    "recovering",
                    state
                        .failure
                        .clone()
                        .unwrap_or_else(|| "搜索服务已退出，等待自动恢复".to_owned()),
                ),
                Probe::Unavailable(detail) => ("unavailable", detail),
            }
        };
        ServiceDiagnostics {
            state: kind.to_owned(),
            pid: state.child.as_ref().map(Child::id),
            detail,
            last_exit: state.last_exit.clone(),
            log_path: self.log_path().to_string_lossy().into_owned(),
            log_tail: log_tail(&self.log_path()),
            restart_attempts: state.attempts.len(),
            build: build_id(),
        }
    }

    pub fn stop(&self) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.stopped = true;
        if let Some(mut child) = state.child.take() {
            self.record(&format!("application exit; stopping pid={}", child.id()));
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    pub fn log_dir(&self) -> PathBuf {
        self.config.data_dir.join("logs")
    }

    fn log_path(&self) -> PathBuf {
        self.log_dir().join("search-core.log")
    }

    fn record(&self, message: &str) {
        let _ = std::fs::create_dir_all(self.log_dir());
        let path = self.log_dir().join("supervisor.log");
        let _ = rotate_log(&path);
        if let Ok(mut log) = OpenOptions::new().create(true).append(true).open(path) {
            let _ = writeln!(log, "{} {message}", now_ms());
        }
    }
}

impl Drop for ServiceSupervisor {
    fn drop(&mut self) {
        self.stop();
    }
}

fn capture_log(mut output: impl Read + Send + 'static, path: PathBuf, lock: Arc<Mutex<()>>) {
    std::thread::spawn(move || {
        let mut buffer = [0_u8; 8192];
        loop {
            let length = match output.read(&mut buffer) {
                Ok(0) => break,
                Ok(length) => length,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => break,
            };
            let _guard = lock.lock().unwrap_or_else(|error| error.into_inner());
            if rotate_log(&path).is_ok() {
                if let Ok(mut log) = OpenOptions::new().create(true).append(true).open(&path) {
                    let _ = log.write_all(&buffer[..length]);
                }
            }
        }
    });
}

fn request_signature(context: &str) -> String {
    let Ok(mut value) = serde_json::from_str::<serde_json::Value>(context) else {
        return context.to_owned();
    };
    if let Some(object) = value.as_object_mut() {
        object.remove("time");
    }
    value.to_string()
}

enum Probe {
    Ready,
    Unresponsive(String),
    Unavailable(String),
}

fn probe(address: SocketAddr) -> Probe {
    let mut stream = match TcpStream::connect_timeout(&address, Duration::from_millis(500)) {
        Ok(stream) => stream,
        Err(error) => {
            return Probe::Unavailable(format!("无法连接本机搜索接口 {address}: {error}"))
        }
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(1)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(1)));
    let request = format!("GET /health HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\n\r\n");
    let response = (|| -> std::io::Result<String> {
        stream.write_all(request.as_bytes())?;
        let mut response = String::new();
        stream.take(16_384).read_to_string(&mut response)?;
        Ok(response)
    })();
    match response {
        Ok(response) if valid_health(&response) => Probe::Ready,
        Ok(_) => Probe::Unresponsive(format!("端口 {address} 可连接，但未返回有效的搜索服务健康响应；可能是服务异常或端口被其他程序占用")),
        Err(error) => Probe::Unresponsive(format!("端口可连接，但健康接口读取失败或超时；不能据此判断服务崩溃，也不会重复启动进程。{error}")),
    }
}

fn valid_health(response: &str) -> bool {
    let Some((headers, body)) = response.split_once("\r\n\r\n") else {
        return false;
    };
    headers
        .lines()
        .next()
        .is_some_and(|line| line.starts_with("HTTP/1.1 200 ") || line.starts_with("HTTP/1.0 200 "))
        && serde_json::from_str::<serde_json::Value>(body)
            .is_ok_and(|value| value["service"] == "search-core" && value["status"] == "ok")
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn build_id() -> String {
    format!(
        "{} / {}",
        env!("CARGO_PKG_VERSION"),
        option_env!("FILESEARCH_BUILD_ID").unwrap_or("development")
    )
}

fn exit_reason(code: Option<i32>) -> String {
    let Some(code) = code else {
        return "进程退出，系统未提供退出码；请查看日志".to_owned();
    };
    let description = match code as u32 {
        0 => "服务自行结束",
        0xC0000005 => "内存访问异常（不等同于内存不足）",
        0xC0000017 => "系统报告内存不足",
        0xC000012D => "系统提交额度或虚拟内存不足",
        0xC000001D => "非法指令，可能与 CPU 指令集兼容性有关",
        0xC00000FD => "线程栈溢出",
        0xC0000135 => "运行所需的 DLL 未找到",
        0xC0000142 => "DLL 初始化失败",
        0xC0000409 => "程序快速失败或安全检查异常，需结合日志定位",
        _ => "服务异常结束，具体原因请查看日志",
    };
    format!("{description}；退出码 {code} (0x{:08X})", code as u32)
}

fn rotate_log(path: &std::path::Path) -> std::io::Result<()> {
    if std::fs::metadata(path).is_ok_and(|metadata| metadata.len() >= LOG_LIMIT) {
        std::fs::copy(path, path.with_extension("previous.log"))?;
        File::create(path)?;
    }
    Ok(())
}

fn log_tail(path: &std::path::Path) -> String {
    let Ok(mut file) = File::open(path) else {
        return String::new();
    };
    let length = file
        .metadata()
        .map(|metadata| metadata.len())
        .unwrap_or_default();
    let _ = file.seek(SeekFrom::Start(length.saturating_sub(16_384)));
    let mut bytes = Vec::new();
    let _ = file.take(16_384).read_to_end(&mut bytes);
    let content = String::from_utf8_lossy(&bytes);
    content
        .lines()
        .rev()
        .take(40)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn health_requires_search_service_identity() {
        assert!(valid_health(
            "HTTP/1.1 200 OK\r\n\r\n{\"service\":\"search-core\",\"status\":\"ok\"}"
        ));
        assert!(!valid_health("HTTP/1.1 200 OK\r\n\r\n{}"));
        assert!(!valid_health(
            "HTTP/1.1 500 Error\r\n\r\n{\"service\":\"search-core\",\"status\":\"ok\"}"
        ));
    }

    #[test]
    fn exit_codes_do_not_mislabel_access_violations_as_out_of_memory() {
        assert!(exit_reason(Some(0xC0000005_u32 as i32)).contains("内存访问异常"));
        assert!(exit_reason(Some(0xC0000017_u32 as i32)).contains("系统报告内存不足"));
        assert!(exit_reason(Some(1)).contains("具体原因请查看日志"));
    }

    #[test]
    fn diagnostics_preserve_utf8_log_tail_and_rotate() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("service.log");
        std::fs::write(&path, "文件解析失败\nmodel error\n").unwrap();
        assert!(log_tail(&path).contains("文件解析失败"));
        let file = OpenOptions::new().write(true).open(&path).unwrap();
        file.set_len(LOG_LIMIT).unwrap();
        drop(file);
        rotate_log(&path).unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 0);
        assert_eq!(
            std::fs::metadata(path.with_extension("previous.log"))
                .unwrap()
                .len(),
            LOG_LIMIT
        );
    }

    fn test_supervisor(directory: &std::path::Path, address: SocketAddr) -> ServiceSupervisor {
        ServiceSupervisor::new(ServiceConfig {
            executable: directory.join("missing-search-core.exe"),
            data_dir: directory.join("data"),
            model_dir: directory.join("models"),
            address,
        })
    }

    fn unused_address() -> SocketAddr {
        std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
    }

    #[test]
    fn startup_failures_have_bounded_retries_and_manual_reset() {
        let directory = tempfile::tempdir().unwrap();
        let supervisor = test_supervisor(directory.path(), unused_address());
        assert!(supervisor.ensure(false).is_err());
        for expected in 1..=MAX_RESTARTS {
            supervisor.state.lock().unwrap().next_retry = None;
            assert!(supervisor.ensure(false).is_err());
            assert_eq!(supervisor.state.lock().unwrap().attempts.len(), expected);
        }
        supervisor.state.lock().unwrap().next_retry = None;
        assert!(supervisor
            .ensure(false)
            .unwrap_err()
            .contains("停止自动重试"));
        assert_eq!(supervisor.diagnostics(None).state, "recovery_failed");
        assert!(supervisor.ensure(true).is_err());
        assert!(supervisor.state.lock().unwrap().attempts.is_empty());
        supervisor.stop();
        assert!(supervisor.ensure(true).unwrap_err().contains("正在退出"));
    }

    #[test]
    fn slow_health_response_does_not_spawn_another_process() {
        let directory = tempfile::tempdir().unwrap();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let supervisor = test_supervisor(directory.path(), listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            let (_stream, _) = listener.accept().unwrap();
            std::thread::sleep(Duration::from_millis(1500));
        });
        let error = supervisor.ensure(false).unwrap_err();
        assert!(error.contains("不会重复启动进程"), "{error}");
        assert!(supervisor.state.lock().unwrap().child.is_none());
        assert!(!supervisor.log_path().exists());
        server.join().unwrap();
    }

    #[test]
    fn repeated_request_timestamps_do_not_defeat_log_throttling() {
        assert_eq!(
            request_signature(r#"{"kind":"timeout","time":1,"path":"/stats"}"#),
            request_signature(r#"{"kind":"timeout","time":2,"path":"/stats"}"#)
        );
    }

    #[test]
    fn streamed_logs_rotate_without_stopping_the_reader() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("service.log");
        let file = File::create(&path).unwrap();
        file.set_len(LOG_LIMIT).unwrap();
        drop(file);
        capture_log(
            std::io::Cursor::new("新的日志\n".as_bytes().to_vec()),
            path.clone(),
            Arc::new(Mutex::new(())),
        );
        let deadline = Instant::now() + Duration::from_secs(3);
        while !log_tail(&path).contains("新的日志") && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(log_tail(&path).contains("新的日志"));
        assert!(path.with_extension("previous.log").exists());
    }

    #[test]
    #[ignore = "requires a locally built search-core executable; uses only a temporary database and random port"]
    fn owned_service_recovers_without_duplicates_and_stops_on_exit() {
        let directory = tempfile::tempdir().unwrap();
        let mut supervisor = test_supervisor(directory.path(), unused_address());
        supervisor.config.executable = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("target/release/search-core.exe");
        assert!(supervisor.config.executable.exists());
        let supervisor = Arc::new(supervisor);
        ServiceSupervisor::monitor(&supervisor);
        supervisor.ensure(false).unwrap();
        let wait_for_ready = || {
            let deadline = Instant::now() + Duration::from_secs(15);
            loop {
                let diagnostic = supervisor.diagnostics(None);
                if diagnostic.state == "running" {
                    return diagnostic;
                }
                assert!(
                    Instant::now() < deadline,
                    "{}\n{}",
                    diagnostic.detail,
                    diagnostic.log_tail
                );
                std::thread::sleep(Duration::from_millis(100));
            }
        };
        let original = wait_for_ready().pid.unwrap();
        supervisor.ensure(false).unwrap();
        assert_eq!(wait_for_ready().pid, Some(original));
        {
            let mut state = supervisor.state.lock().unwrap();
            let child = state.child.as_mut().unwrap();
            child.kill().unwrap();
            child.wait().unwrap();
        }
        supervisor.diagnostics(None);
        let recovered = wait_for_ready();
        assert_ne!(recovered.pid, Some(original));
        assert!(recovered.last_exit.is_some());
        assert_eq!(recovered.restart_attempts, 1);
        assert!(log_tail(&supervisor.log_dir().join("supervisor.log")).contains("exited pid="));
        supervisor.stop();
        assert!(matches!(
            probe(supervisor.config.address),
            Probe::Unavailable(_)
        ));
        assert_eq!(supervisor.diagnostics(None).state, "stopped");
        assert!(supervisor.ensure(true).is_err());
    }
}
