use crate::extract::is_temporary_office_file;
use anyhow::Result;
use notify::event::{MetadataKind, ModifyKind};
use notify::{Config, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const DEBOUNCE: Duration = Duration::from_millis(900);
const ROOT_HEALTH_INTERVAL: Duration = Duration::from_secs(30);

pub struct WatchService {
    _watcher: RecommendedWatcher,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    watched_roots: usize,
}

impl WatchService {
    pub fn start<F, E, R>(
        roots: Vec<PathBuf>,
        mut on_paths: F,
        mut on_error: E,
        mut on_recovered: R,
    ) -> Result<Self>
    where
        F: FnMut(Vec<PathBuf>) + Send + 'static,
        E: FnMut(String) + Send + 'static,
        R: FnMut(PathBuf) + Send + 'static,
    {
        let (sender, receiver) = mpsc::channel();
        let mut watcher = RecommendedWatcher::new(
            move |event| {
                let _ = sender.send(event);
            },
            Config::default(),
        )?;
        let mut watched_roots = 0usize;
        for root in &roots {
            match watcher.watch(root, RecursiveMode::Recursive) {
                Ok(()) => watched_roots += 1,
                Err(error) => on_error(format!("无法监听 {}: {error}", root.display())),
            }
        }

        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let worker = thread::spawn(move || {
            let mut pending = HashSet::new();
            let mut last_event = Instant::now();
            let mut last_health_check = Instant::now();
            let mut root_health = roots
                .iter()
                .map(|root| (root.clone(), root.is_dir()))
                .collect::<HashMap<_, _>>();
            while !worker_stop.load(Ordering::Relaxed) {
                match receiver.recv_timeout(Duration::from_millis(200)) {
                    Ok(Ok(event)) => {
                        if event_affects_index(&event) {
                            pending.extend(
                                event
                                    .paths
                                    .into_iter()
                                    .filter(|path| !is_temporary_office_file(path)),
                            );
                            if !pending.is_empty() {
                                last_event = Instant::now();
                            }
                        }
                    }
                    Ok(Err(error)) => on_error(format!("文件监听异常: {error}")),
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                }
                if !pending.is_empty() && last_event.elapsed() >= DEBOUNCE {
                    on_paths(pending.drain().collect());
                }
                if last_health_check.elapsed() >= ROOT_HEALTH_INTERVAL {
                    for (root, was_online) in &mut root_health {
                        let is_online = root.is_dir();
                        if *was_online && !is_online {
                            on_error(format!("共享盘或索引目录已断线: {}", root.display()));
                        } else if !*was_online && is_online {
                            // The original OS watch can be invalid after a mount
                            // disappears. Let the application re-index this root
                            // and recreate the watcher once it is available.
                            on_recovered(root.clone());
                        }
                        *was_online = is_online;
                    }
                    last_health_check = Instant::now();
                }
            }
        });
        Ok(Self {
            _watcher: watcher,
            stop,
            worker: Some(worker),
            watched_roots,
        })
    }

    pub fn status(&self) -> String {
        format!("active ({})", self.watched_roots)
    }
}

fn event_affects_index(event: &Event) -> bool {
    !matches!(
        event.kind,
        EventKind::Access(_) | EventKind::Modify(ModifyKind::Metadata(MetadataKind::AccessTime))
    )
}

impl Drop for WatchService {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use notify::event::{AccessKind, AccessMode};

    #[test]
    fn ignores_file_access_and_access_time_events() {
        let opened = Event::new(EventKind::Access(AccessKind::Open(AccessMode::Read)));
        let access_time = Event::new(EventKind::Modify(ModifyKind::Metadata(
            MetadataKind::AccessTime,
        )));
        assert!(!event_affects_index(&opened));
        assert!(!event_affects_index(&access_time));
    }

    #[test]
    fn keeps_content_create_rename_and_remove_events() {
        assert!(event_affects_index(&Event::new(EventKind::Create(
            notify::event::CreateKind::File,
        ))));
        assert!(event_affects_index(&Event::new(EventKind::Modify(
            ModifyKind::Data(notify::event::DataChange::Content),
        ))));
        assert!(event_affects_index(&Event::new(EventKind::Remove(
            notify::event::RemoveKind::File,
        ))));
    }
}
