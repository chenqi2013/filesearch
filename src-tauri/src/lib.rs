use std::path::Path;
use std::process::Command as SystemCommand;
use std::sync::Mutex;
use std::time::Duration;
use tauri::{AppHandle, Manager, RunEvent, State};
use tauri_plugin_shell::process::CommandChild;
use tauri_plugin_shell::ShellExt;

const CORE_PORT: u16 = 47653;

#[derive(Default)]
struct CoreProcess(Mutex<Option<CommandChild>>);

#[tauri::command]
async fn ensure_search_core(app: AppHandle, state: State<'_, CoreProcess>) -> Result<u16, String> {
    if core_is_ready().await {
        return Ok(CORE_PORT);
    }
    let data_dir = app
        .path()
        .app_local_data_dir()
        .map_err(|error| error.to_string())?;
    std::fs::create_dir_all(&data_dir).map_err(|error| error.to_string())?;
    let (mut events, child) = app
        .shell()
        .sidecar("search-core")
        .map_err(|error| error.to_string())?
        .args([
            "--listen",
            "127.0.0.1:47653",
            "--data-dir",
            &data_dir.to_string_lossy(),
        ])
        .spawn()
        .map_err(|error| error.to_string())?;
    *state.0.lock().map_err(|_| "搜索核心进程锁异常")? = Some(child);
    tauri::async_runtime::spawn(async move { while events.recv().await.is_some() {} });
    for _ in 0..50 {
        if core_is_ready().await {
            return Ok(CORE_PORT);
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    Err("搜索核心启动超时".to_owned())
}

async fn core_is_ready() -> bool {
    tokio::net::TcpStream::connect(("127.0.0.1", CORE_PORT))
        .await
        .is_ok()
}

#[tauri::command]
fn open_path(path: String) -> Result<(), String> {
    ensure_exists(&path)?;
    run_open_command(&path, false)
}

#[tauri::command]
fn reveal_path(path: String) -> Result<(), String> {
    ensure_exists(&path)?;
    run_open_command(&path, true)
}

fn ensure_exists(path: &str) -> Result<(), String> {
    if Path::new(path).exists() {
        Ok(())
    } else {
        Err("文件不存在或已被移动".to_owned())
    }
}

fn run_open_command(path: &str, reveal: bool) -> Result<(), String> {
    #[cfg(target_os = "windows")]
    let mut command = {
        let mut command = SystemCommand::new("explorer.exe");
        if reveal {
            command.arg(format!("/select,{path}"));
        } else {
            command.arg(path);
        }
        command
    };
    #[cfg(target_os = "macos")]
    let mut command = {
        let mut command = SystemCommand::new("open");
        if reveal {
            command.arg("-R");
        }
        command.arg(path);
        command
    };
    #[cfg(all(unix, not(target_os = "macos")))]
    let mut command = {
        let target = if reveal {
            Path::new(path).parent().unwrap_or_else(|| Path::new(path))
        } else {
            Path::new(path)
        };
        let mut command = SystemCommand::new("xdg-open");
        command.arg(target);
        command
    };
    command
        .spawn()
        .map(|_| ())
        .map_err(|error| error.to_string())
}

pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .manage(CoreProcess::default())
        .invoke_handler(tauri::generate_handler![
            ensure_search_core,
            open_path,
            reveal_path
        ])
        .build(tauri::generate_context!())
        .expect("error while building Tauri application");
    app.run(|handle, event| {
        if matches!(event, RunEvent::Exit) {
            if let Ok(mut process) = handle.state::<CoreProcess>().0.lock() {
                if let Some(child) = process.take() {
                    let _ = child.kill();
                }
            }
        }
    });
}
