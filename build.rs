// Compile our packaged, decoder-only C sources directly into the Rust library.
// The source lists and configuration headers travel with the crate.
use std::{
    env, fs,
    path::{Path, PathBuf},
};

fn paths(root: &Path, manifest: &str) -> Vec<PathBuf> {
    let list = fs::read_to_string(root.join(manifest)).expect("missing AVIF source manifest");
    list.lines()
        .map(str::trim)
        .filter(|s| !s.is_empty() && !s.starts_with('#'))
        .map(|s| root.join(s))
        .collect()
}

fn main() {
    let root = Path::new("vendor/avif");
    println!("cargo:rerun-if-changed={}", root.display());
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let mut build = cc::Build::new();
    build
        .std("c11")
        .warnings(false)
        .define("NDEBUG", None)
        .files(paths(root, "sources.txt"));
    if env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        build.include(root.join("dav1d/include/compat/msvc"));
    }
    build.includes(paths(root, "includes.txt"));
    if env::var("CARGO_CFG_TARGET_VENDOR").as_deref() == Ok("apple") {
        build.define("_DARWIN_C_SOURCE", None);
    }
    if matches!(os.as_str(), "linux" | "android") {
        build.define("_GNU_SOURCE", None);
    }
    if os != "windows" {
        build.flag_if_supported("-pthread");
        println!("cargo:rustc-link-lib=m");
    }
    build.compile("nanoodle_avif");
}
