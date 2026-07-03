use std::env;
use std::fmt::Write as _;
use std::path::PathBuf;

const SUNNY_WIDTH: usize = 480;
const SUNNY_HEIGHT: usize = 600;
const SUNNY_BYTES: u64 = (SUNNY_WIDTH * SUNNY_HEIGHT * 4) as u64;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(has_sunny_cat)");
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set"));
    let generated_path = out_dir.join("sunny_cat_asset.rs");
    let mut generated = String::new();

    let candidates = sunny_candidates();
    for candidate in &candidates {
        println!("cargo:rerun-if-changed={}", candidate.display());
    }
    println!("cargo:rerun-if-env-changed=HAZE_CGEN_SUNNY_RGBA");

    let asset = candidates.into_iter().find(|path| {
        path.metadata()
            .map(|metadata| metadata.is_file() && metadata.len() == SUNNY_BYTES)
            .unwrap_or(false)
    });

    if let Some(path) = asset {
        let path = path.canonicalize().unwrap_or(path);
        println!("cargo:rustc-cfg=has_sunny_cat");
        let _ = writeln!(generated, "pub(crate) const AVAILABLE: bool = true;");
        let _ = writeln!(generated, "pub(crate) const WIDTH: usize = {SUNNY_WIDTH};");
        let _ = writeln!(
            generated,
            "pub(crate) const HEIGHT: usize = {SUNNY_HEIGHT};"
        );
        let _ = writeln!(
            generated,
            "pub(crate) const RGBA: &[u8] = include_bytes!({:?});",
            path.display().to_string()
        );
    } else {
        let _ = writeln!(generated, "pub(crate) const AVAILABLE: bool = false;");
        let _ = writeln!(generated, "pub(crate) const WIDTH: usize = {SUNNY_WIDTH};");
        let _ = writeln!(
            generated,
            "pub(crate) const HEIGHT: usize = {SUNNY_HEIGHT};"
        );
        let _ = writeln!(generated, "pub(crate) const RGBA: &[u8] = &[];");
    }

    std::fs::write(generated_path, generated).expect("write generated sunny cat asset");
}

fn sunny_candidates() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(path) = env::var_os("HAZE_CGEN_SUNNY_RGBA") {
        paths.push(PathBuf::from(path));
    }
    if let Some(profile) = env::var_os("USERPROFILE") {
        paths.push(PathBuf::from(profile).join("Pictures").join("sunny.rgba"));
    }
    paths.push(PathBuf::from(r"C:\Users\rai\Pictures\sunny.rgba"));
    paths
}
