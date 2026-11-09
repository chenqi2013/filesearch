//! Keep transient Windows file-sharing conflicts from poisoning a Tantivy writer.
use std::{io, path::Path, sync::Arc, time::Duration};
use tantivy::directory::{
    error::{DeleteError, LockError, OpenReadError, OpenWriteError},
    Directory, DirectoryLock, FileHandle, Lock, MmapDirectory, WatchCallback, WatchHandle,
    WritePtr,
};

#[derive(Clone, Debug)]
pub struct IndexDirectory(pub MmapDirectory);

fn retry_open<T>(mut open: impl FnMut() -> Result<T, OpenWriteError>) -> Result<T, OpenWriteError> {
    for attempt in 0..=5 {
        match open() {
            Err(OpenWriteError::IoError { ref io_error, .. })
                if cfg!(windows)
                    && matches!(io_error.raw_os_error(), Some(5 | 32 | 33))
                    && attempt < 5 =>
            {
                // Never remove/truncate an existing file, or retry an AlreadyExists error.
                tracing::warn!(attempt, error = %io_error, "retrying index file creation after sharing conflict");
                std::thread::sleep(Duration::from_millis(20 << attempt));
            }
            result => return result,
        }
    }
    unreachable!()
}

impl Directory for IndexDirectory {
    fn get_file_handle(&self, path: &Path) -> Result<Arc<dyn FileHandle>, OpenReadError> {
        self.0.get_file_handle(path)
    }
    fn delete(&self, path: &Path) -> Result<(), DeleteError> {
        self.0.delete(path)
    }
    fn exists(&self, path: &Path) -> Result<bool, OpenReadError> {
        self.0.exists(path)
    }
    fn open_write(&self, path: &Path) -> Result<WritePtr, OpenWriteError> {
        retry_open(|| self.0.open_write(path))
    }
    fn atomic_read(&self, path: &Path) -> Result<Vec<u8>, OpenReadError> {
        self.0.atomic_read(path)
    }
    fn atomic_write(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        // ManagedDirectory also rewrites .managed.json before creating each segment file.
        // A scanner temporarily opening that metadata file can prevent Windows rename.
        for attempt in 0..=5 {
            match self.0.atomic_write(path, data) {
                Err(error)
                    if cfg!(windows)
                        && matches!(error.raw_os_error(), Some(5 | 32 | 33))
                        && attempt < 5 =>
                {
                    tracing::warn!(attempt, path = %path.display(), %error, "retrying atomic index metadata write");
                    std::thread::sleep(Duration::from_millis(20 << attempt));
                }
                result => return result,
            }
        }
        unreachable!()
    }
    fn sync_directory(&self) -> io::Result<()> {
        self.0.sync_directory()
    }
    fn acquire_lock(&self, lock: &Lock) -> Result<DirectoryLock, LockError> {
        self.0.acquire_lock(lock)
    }
    fn watch(&self, callback: WatchCallback) -> tantivy::Result<WatchHandle> {
        self.0.watch(callback)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[cfg(windows)]
    fn atomic_metadata_write_recovers_from_temporary_exclusive_handle() {
        use std::os::windows::fs::OpenOptionsExt;
        let temp = tempfile::tempdir().unwrap();
        let directory = IndexDirectory(MmapDirectory::open(temp.path()).unwrap());
        directory
            .atomic_write(Path::new(".managed.json"), b"old")
            .unwrap();
        let locked = std::fs::OpenOptions::new()
            .read(true)
            .share_mode(0)
            .open(temp.path().join(".managed.json"))
            .unwrap();
        let copy = directory.clone();
        let write =
            std::thread::spawn(move || copy.atomic_write(Path::new(".managed.json"), b"new"));
        std::thread::sleep(Duration::from_millis(45));
        drop(locked);
        write.join().unwrap().unwrap();
        assert_eq!(
            directory.atomic_read(Path::new(".managed.json")).unwrap(),
            b"new"
        );
    }
    #[test]
    fn retries_transient_creation_but_preserves_existing_files() {
        let mut attempts = 0;
        let result = retry_open(|| {
            attempts += 1;
            if cfg!(windows) && attempts < 3 {
                Err(OpenWriteError::wrap_io_error(
                    io::Error::from_raw_os_error(32),
                    "test.pos".into(),
                ))
            } else {
                Ok(17)
            }
        })
        .unwrap();
        assert_eq!(result, 17);
        assert_eq!(attempts, if cfg!(windows) { 3 } else { 1 });
        let mut attempts = 0;
        let result: Result<(), _> = retry_open(|| {
            attempts += 1;
            Err(OpenWriteError::FileAlreadyExists("existing.pos".into()))
        });
        assert!(matches!(result, Err(OpenWriteError::FileAlreadyExists(_))));
        assert_eq!(attempts, 1);
    }
}
