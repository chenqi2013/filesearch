mod service;

use service::{ServiceConfig, ServiceDiagnostics, ServiceSupervisor};
use std::collections::HashMap;
use std::path::Path;
use std::process::Command as SystemCommand;
use std::sync::Arc;
use std::time::{Duration, UNIX_EPOCH};
use tauri::menu::{Menu, MenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Manager, RunEvent, State};

const CORE_PORT: u16 = 47653;
const LIBREOFFICE_DOWNLOAD_URL: &str = "https://www.libreoffice.org/download/download-libreoffice/";

struct CoreProcess(Arc<ServiceSupervisor>);

#[tauri::command]
async fn ensure_search_core(
    state: State<'_, CoreProcess>,
    retry: Option<bool>,
) -> Result<u16, String> {
    let supervisor = Arc::clone(&state.0);
    tauri::async_runtime::spawn_blocking(move || supervisor.ensure(retry.unwrap_or(false)))
        .await
        .map_err(|error| error.to_string())??;
    for _ in 0..50 {
        let supervisor = Arc::clone(&state.0);
        let diagnostics =
            tauri::async_runtime::spawn_blocking(move || supervisor.diagnostics(None))
                .await
                .map_err(|error| error.to_string())?;
        if diagnostics.state == "running" {
            return Ok(CORE_PORT);
        }
        if matches!(
            diagnostics.state.as_str(),
            "recovering" | "recovery_failed" | "stopped" | "unresponsive"
        ) {
            return Err(diagnostics.detail);
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    Err("搜索核心启动超时".to_owned())
}

#[tauri::command]
async fn search_core_diagnostics(
    state: State<'_, CoreProcess>,
    context: Option<String>,
) -> Result<ServiceDiagnostics, String> {
    let supervisor = Arc::clone(&state.0);
    tauri::async_runtime::spawn_blocking(move || supervisor.diagnostics(context.as_deref()))
        .await
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn open_service_logs(state: State<'_, CoreProcess>) -> Result<(), String> {
    let directory = state.0.log_dir();
    std::fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
    run_open_command(&directory.to_string_lossy(), false)
}

#[tauri::command]
fn hide_to_tray(app: AppHandle) -> Result<(), String> {
    app.get_webview_window("main")
        .ok_or("无法定位主窗口")?
        .hide()
        .map_err(|error| error.to_string())
}

fn show_main_window(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
    }
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

#[tauri::command]
fn open_libreoffice_download() -> Result<(), String> {
    run_open_url_command(LIBREOFFICE_DOWNLOAD_URL)
}

#[tauri::command]
fn directory_modified_times(paths: Vec<String>) -> HashMap<String, u64> {
    paths
        .into_iter()
        .map(|path| {
            let modified_ms = std::fs::metadata(&path)
                .and_then(|metadata| metadata.modified())
                .ok()
                .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
                .map(|duration| duration.as_millis() as u64)
                .unwrap_or_default();
            (path, modified_ms)
        })
        .collect()
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

fn run_open_url_command(url: &str) -> Result<(), String> {
    #[cfg(target_os = "windows")]
    let mut command = {
        let mut command = SystemCommand::new("rundll32.exe");
        command.args(["url.dll,FileProtocolHandler", url]);
        command
    };
    #[cfg(target_os = "macos")]
    let mut command = {
        let mut command = SystemCommand::new("open");
        command.arg(url);
        command
    };
    #[cfg(all(unix, not(target_os = "macos")))]
    let mut command = {
        let mut command = SystemCommand::new("xdg-open");
        command.arg(url);
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
        .setup(|app| {
            let data_dir = app.path().app_local_data_dir()?;
            let executable = std::env::current_exe()?
                .parent()
                .ok_or("无法定位搜索服务目录")?
                .join(if cfg!(windows) {
                    "search-core.exe"
                } else {
                    "search-core"
                });
            let supervisor = Arc::new(ServiceSupervisor::new(ServiceConfig {
                executable,
                data_dir,
                model_dir: app
                    .path()
                    .resource_dir()?
                    .join("models")
                    .join("embedding-rwkv-tiny"),
                address: format!("127.0.0.1:{CORE_PORT}").parse()?,
            }));
            ServiceSupervisor::monitor(&supervisor);
            app.manage(CoreProcess(supervisor));
            let show = MenuItem::with_id(app, "show", "显示主窗口", true, None::<&str>)?;
            let logs = MenuItem::with_id(app, "logs", "打开诊断日志", true, None::<&str>)?;
            let exit =
                MenuItem::with_id(app, "exit", "退出本地知搜（停止索引）", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&show, &logs, &exit])?;
            let mut tray = TrayIconBuilder::new()
                .tooltip("本地知搜 — 关闭窗口后继续在后台运行")
                .menu(&menu)
                .show_menu_on_left_click(false)
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "show" => show_main_window(app),
                    "logs" => {
                        let _ = open_service_logs(app.state::<CoreProcess>());
                    }
                    "exit" => {
                        app.state::<CoreProcess>().0.stop();
                        app.exit(0);
                    }
                    _ => {}
                })
                .on_tray_icon_event(|tray, event| {
                    if matches!(
                        event,
                        TrayIconEvent::Click {
                            button: MouseButton::Left,
                            button_state: MouseButtonState::Up,
                            ..
                        }
                    ) {
                        show_main_window(tray.app_handle());
                    }
                });
            if let Some(icon) = app.default_window_icon() {
                tray = tray.icon(icon.clone());
            }
            tray.build(app)?;
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                match window.hide() {
                    Ok(()) => api.prevent_close(),
                    Err(_) => window.app_handle().state::<CoreProcess>().0.stop(),
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            ensure_search_core,
            search_core_diagnostics,
            open_service_logs,
            hide_to_tray,
            open_path,
            reveal_path,
            open_libreoffice_download,
            directory_modified_times
        ])
        .build(tauri::generate_context!())
        .expect("error while building Tauri application");
    app.run(|handle, event| {
        if matches!(event, RunEvent::Exit) {
            handle.state::<CoreProcess>().0.stop();
        }
    });
}
