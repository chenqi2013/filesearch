fn main() {
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=../src");
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    println!("cargo:rustc-env=FILESEARCH_BUILD_ID={timestamp}");
    tauri_build::build()
}
