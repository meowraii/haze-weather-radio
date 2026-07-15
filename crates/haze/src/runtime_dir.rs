use std::ffi::OsStr;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

mod default_assets {
    include!(concat!(env!("OUT_DIR"), "/default_assets.rs"));
}

const RUNTIME_MARKER: &str = ".haze-runtime";
const DEFAULT_CHILD_DIR: &str = "haze-runtime";
const RUNTIME_DIRS: &[&str] = &["audio", "data", "logs", "managed", "runtime"];
const APP_FILES: &[&str] = &["config.yaml"];
const APP_DIRS: &[&str] = &["webroot"];
const MANAGED_DIRS: &[&str] = &["configs", "csv", "voices"];
const MANAGED_FILES: &[&str] = &[
    "dictionary.json",
    "sameMapping.json",
    "staticPhrases.json",
    "userbulletins.json",
];
const IGNORED_NAMES: &[&str] = &[
    ".DS_Store",
    ".env",
    ".env.local",
    ".git",
    ".gitattributes",
    ".gitignore",
    ".haze-runtime",
    "Thumbs.db",
    "haze",
    "haze.cmd",
    "haze.exe",
    "target",
];

pub(crate) struct RuntimeLayout {
    pub app_dir: PathBuf,
    pub runtime_dir: PathBuf,
}

pub(crate) fn resolve(explicit_runtime_dir: Option<&Path>) -> Result<RuntimeLayout> {
    let app_dir = resolve_app_dir()?;
    let runtime_dir = match explicit_runtime_dir {
        Some(path) => canonicalize_or_create(path)?,
        None => choose_default_runtime_dir(&app_dir)?,
    };

    prepare_runtime_dir(&app_dir, &runtime_dir)?;

    Ok(RuntimeLayout {
        app_dir,
        runtime_dir,
    })
}

fn resolve_app_dir() -> Result<PathBuf> {
    let executable = std::env::current_exe().context("failed to locate Haze executable")?;
    if let Some(parent) = executable.parent() {
        if is_app_dir(parent) {
            return canonicalize_normal(parent).with_context(|| {
                format!(
                    "failed to resolve executable directory {}",
                    parent.display()
                )
            });
        }
    }

    let current = std::env::current_dir().context("failed to read current working directory")?;
    if is_app_dir(&current) {
        return canonicalize_normal(&current)
            .with_context(|| format!("failed to resolve current directory {}", current.display()));
    }

    if let Some(parent) = executable.parent() {
        if is_executable_dir(parent) {
            return canonicalize_normal(parent).with_context(|| {
                format!(
                    "failed to resolve executable directory {}",
                    parent.display()
                )
            });
        }
    }

    bail!("could not find Haze host bundle files; run from the bundle directory or pass --workdir");
}

fn is_app_dir(path: &Path) -> bool {
    path.join("config.yaml").is_file()
        && (path.join("webroot").is_dir() || path.join("managed").is_dir())
}

fn is_executable_dir(path: &Path) -> bool {
    let Ok(executable) = std::env::current_exe() else {
        return false;
    };
    executable.parent() == Some(path)
}

fn choose_default_runtime_dir(app_dir: &Path) -> Result<PathBuf> {
    if should_prompt_for_runtime_dir(app_dir)? {
        return prompt_runtime_dir(app_dir);
    }
    canonicalize_or_create(app_dir)
}

fn prompt_runtime_dir(app_dir: &Path) -> Result<PathBuf> {
    eprintln!();
    eprintln!(
        "Haze is about to create runtime files such as audio, data, logs, and service state."
    );
    eprintln!("The executable directory is not empty:");
    eprintln!("  {}", friendly_path(app_dir));
    eprintln!();
    eprintln!("Choose where Haze should keep runtime files:");
    eprintln!("  1) Continue here");
    eprintln!("  2) Create/use ./{}", DEFAULT_CHILD_DIR);
    eprintln!("  3) Choose a custom directory");
    eprintln!("  q) Cancel");

    loop {
        let choice = prompt_line("Selection [1/2/3/q]: ")?;
        match choice.trim().to_ascii_lowercase().as_str() {
            "1" | "c" | "continue" | "here" => {
                let runtime_dir = canonicalize_or_create(app_dir)?;
                mark_runtime_dir(&runtime_dir)?;
                return Ok(runtime_dir);
            }
            "2" | "n" | "new" => {
                let runtime_dir = next_child_runtime_dir(app_dir);
                fs::create_dir_all(&runtime_dir).with_context(|| {
                    format!(
                        "failed to create runtime directory {}",
                        runtime_dir.display()
                    )
                })?;
                let runtime_dir = canonicalize_normal(&runtime_dir).with_context(|| {
                    format!(
                        "failed to resolve runtime directory {}",
                        runtime_dir.display()
                    )
                })?;
                mark_runtime_dir(&runtime_dir)?;
                return Ok(runtime_dir);
            }
            "3" | "custom" => {
                let raw = prompt_line("Runtime directory: ")?;
                let trimmed = raw.trim();
                if trimmed.is_empty() {
                    eprintln!("Please enter a directory path.");
                    continue;
                }
                let runtime_dir = resolve_custom_path(trimmed)?;
                if should_prompt_for_runtime_dir(&runtime_dir)?
                    && !confirm_nonempty_custom_dir(&runtime_dir)?
                {
                    continue;
                }
                let runtime_dir = canonicalize_or_create(&runtime_dir)?;
                mark_runtime_dir(&runtime_dir)?;
                return Ok(runtime_dir);
            }
            "q" | "quit" | "cancel" => bail!("runtime directory selection cancelled"),
            _ => eprintln!("Please choose 1, 2, 3, or q."),
        }
    }
}

fn confirm_nonempty_custom_dir(path: &Path) -> Result<bool> {
    eprintln!("That directory is not empty:");
    eprintln!("  {}", path.display());
    loop {
        let choice = prompt_line("Use it anyway? [y/N]: ")?;
        match choice.trim().to_ascii_lowercase().as_str() {
            "y" | "yes" => return Ok(true),
            "" | "n" | "no" => return Ok(false),
            _ => eprintln!("Please answer y or n."),
        }
    }
}

fn prompt_line(prompt: &str) -> Result<String> {
    eprint!("{prompt}");
    io::stderr().flush().context("failed to flush prompt")?;

    let mut value = String::new();
    let read = io::stdin()
        .read_line(&mut value)
        .context("failed to read terminal input")?;
    if read == 0 {
        bail!("no terminal input available for runtime directory selection");
    }
    Ok(value)
}

fn friendly_path(path: &Path) -> String {
    dunce::simplified(path).display().to_string()
}

fn resolve_custom_path(raw: &str) -> Result<PathBuf> {
    let candidate = PathBuf::from(raw.trim_matches('"'));
    if candidate.is_absolute() {
        return Ok(dunce::simplified(&candidate).to_path_buf());
    }
    let current = std::env::current_dir().context("failed to read current working directory")?;
    Ok(dunce::simplified(&current.join(candidate)).to_path_buf())
}

fn canonicalize_or_create(path: &Path) -> Result<PathBuf> {
    fs::create_dir_all(path)
        .with_context(|| format!("failed to create runtime directory {}", path.display()))?;
    canonicalize_normal(path)
        .with_context(|| format!("failed to resolve runtime directory {}", path.display()))
}

fn canonicalize_normal(path: &Path) -> io::Result<PathBuf> {
    path.canonicalize()
        .map(|canonical| dunce::simplified(&canonical).to_path_buf())
}

fn mark_runtime_dir(path: &Path) -> Result<()> {
    fs::write(
        path.join(RUNTIME_MARKER),
        "Haze Weather Radio runtime directory\n",
    )
    .with_context(|| format!("failed to mark runtime directory {}", path.display()))
}

fn prepare_runtime_dir(app_dir: &Path, runtime_dir: &Path) -> Result<()> {
    for directory in RUNTIME_DIRS {
        fs::create_dir_all(runtime_dir.join(directory)).with_context(|| {
            format!(
                "failed to create runtime directory {}",
                runtime_dir.join(directory).display()
            )
        })?;
    }
    fs::create_dir_all(runtime_dir.join("data").join("alerts")).with_context(|| {
        format!(
            "failed to create runtime directory {}",
            runtime_dir.join("data").join("alerts").display()
        )
    })?;
    for directory in [
        ["runtime", "audio", "alerts"].as_slice(),
        ["runtime", "audio", "playlist"].as_slice(),
        ["runtime", "audio", "playout"].as_slice(),
        ["runtime", "audio", "tts"].as_slice(),
        ["runtime", "feeds"].as_slice(),
        ["runtime", "playlists"].as_slice(),
        ["runtime", "products"].as_slice(),
        ["runtime", "queues", "alerts"].as_slice(),
        ["runtime", "services", "bin"].as_slice(),
        ["runtime", "state"].as_slice(),
    ] {
        let path = directory
            .iter()
            .fold(runtime_dir.to_path_buf(), |path, part| path.join(part));
        fs::create_dir_all(&path)
            .with_context(|| format!("failed to create runtime directory {}", path.display()))?;
    }

    if !same_path(app_dir, runtime_dir) {
        seed_host_app_files(app_dir, runtime_dir)?;
        seed_audio_files(app_dir, runtime_dir)?;
    }
    seed_embedded_defaults(runtime_dir)?;
    Ok(())
}

fn seed_embedded_defaults(runtime_dir: &Path) -> Result<()> {
    for (relative, bytes) in default_assets::DEFAULT_ASSETS {
        let target = safe_asset_target(runtime_dir, relative)?;
        if target.exists() {
            continue;
        }
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create directory {}", parent.display()))?;
        }
        fs::write(&target, bytes)
            .with_context(|| format!("failed to write default asset {}", target.display()))?;
    }
    Ok(())
}

fn safe_asset_target(runtime_dir: &Path, relative: &str) -> Result<PathBuf> {
    let relative_path = Path::new(relative);
    if relative_path.is_absolute()
        || relative_path
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir))
    {
        bail!("embedded default asset has unsafe path {relative}");
    }
    Ok(runtime_dir.join(relative_path))
}

fn seed_host_app_files(app_dir: &Path, runtime_dir: &Path) -> Result<()> {
    for file_name in APP_FILES {
        copy_file_if_missing(&app_dir.join(file_name), &runtime_dir.join(file_name))?;
    }

    for dir_name in APP_DIRS {
        sync_app_dir(&app_dir.join(dir_name), &runtime_dir.join(dir_name))?;
    }
    seed_managed_files(app_dir, runtime_dir)?;

    Ok(())
}

fn seed_managed_files(app_dir: &Path, runtime_dir: &Path) -> Result<()> {
    let source_managed = app_dir.join("managed");
    let target_managed = runtime_dir.join("managed");
    if !source_managed.is_dir() {
        return Ok(());
    }
    fs::create_dir_all(&target_managed).with_context(|| {
        format!(
            "failed to create managed directory {}",
            target_managed.display()
        )
    })?;
    for dir_name in MANAGED_DIRS {
        copy_dir_if_missing(
            &source_managed.join(dir_name),
            &target_managed.join(dir_name),
        )?;
    }
    for file_name in MANAGED_FILES {
        copy_file_if_missing(
            &source_managed.join(file_name),
            &target_managed.join(file_name),
        )?;
    }
    Ok(())
}

fn seed_audio_files(app_dir: &Path, runtime_dir: &Path) -> Result<()> {
    let source_audio = app_dir.join("audio");
    if !source_audio.is_dir() {
        return Ok(());
    }

    let target_audio = runtime_dir.join("audio");
    fs::create_dir_all(&target_audio).with_context(|| {
        format!(
            "failed to create audio directory {}",
            target_audio.display()
        )
    })?;

    for entry in fs::read_dir(&source_audio)
        .with_context(|| format!("failed to read audio directory {}", source_audio.display()))?
    {
        let entry = entry.with_context(|| format!("failed to read {}", source_audio.display()))?;
        let file_type = entry
            .file_type()
            .with_context(|| format!("failed to inspect audio entry {}", entry.path().display()))?;
        let target = target_audio.join(entry.file_name());
        if file_type.is_file() {
            copy_file_if_missing(&entry.path(), &target)?;
        } else if file_type.is_dir() && entry.file_name() == OsStr::new("static") {
            copy_dir_if_missing(&entry.path(), &target)?;
        }
    }

    Ok(())
}

fn copy_file_if_missing(source: &Path, target: &Path) -> Result<()> {
    if target.exists() || !source.is_file() {
        return Ok(());
    }
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create directory {}", parent.display()))?;
    }
    fs::copy(source, target).with_context(|| {
        format!(
            "failed to copy {} to {}",
            source.display(),
            target.display()
        )
    })?;
    Ok(())
}

fn copy_dir_if_missing(source: &Path, target: &Path) -> Result<()> {
    if target.exists() || !source.is_dir() {
        return Ok(());
    }
    copy_dir_recursive(source, target)
}

fn sync_app_dir(source: &Path, target: &Path) -> Result<()> {
    if !source.is_dir() {
        return Ok(());
    }
    if target.exists() {
        fs::remove_dir_all(target)
            .with_context(|| format!("failed to replace app directory {}", target.display()))?;
    }
    copy_dir_recursive(source, target)
}

fn copy_dir_recursive(source: &Path, target: &Path) -> Result<()> {
    fs::create_dir_all(target)
        .with_context(|| format!("failed to create directory {}", target.display()))?;
    for entry in
        fs::read_dir(source).with_context(|| format!("failed to read {}", source.display()))?
    {
        let entry = entry.with_context(|| format!("failed to read {}", source.display()))?;
        let source_path = entry.path();
        let target_path = target.join(entry.file_name());
        let file_type = entry
            .file_type()
            .with_context(|| format!("failed to inspect {}", source_path.display()))?;
        if file_type.is_dir() {
            copy_dir_recursive(&source_path, &target_path)?;
        } else if file_type.is_file() {
            copy_file_if_missing(&source_path, &target_path)?;
        }
    }
    Ok(())
}

fn same_path(left: &Path, right: &Path) -> bool {
    match (canonicalize_normal(left), canonicalize_normal(right)) {
        (Ok(left), Ok(right)) => left == right,
        _ => left == right,
    }
}

pub(crate) fn configured_runtime_relative_path(
    app_dir: &Path,
    runtime_dir: &Path,
    configured_path: &Path,
) -> PathBuf {
    if app_dir != runtime_dir {
        if let Ok(relative) = configured_path.strip_prefix("runtime") {
            return relative.to_path_buf();
        }
    }
    configured_path.to_path_buf()
}

pub(crate) fn resolve_configured_runtime_path(
    app_dir: &Path,
    runtime_dir: &Path,
    configured_path: &Path,
) -> PathBuf {
    let relative = configured_runtime_relative_path(app_dir, runtime_dir, configured_path);
    if relative.is_absolute() {
        relative
    } else {
        runtime_dir.join(relative)
    }
}

fn should_prompt_for_runtime_dir(path: &Path) -> Result<bool> {
    if !path.exists() {
        return Ok(false);
    }
    if path.join(RUNTIME_MARKER).is_file() {
        return Ok(false);
    }
    has_meaningful_entries(path)
}

fn has_meaningful_entries(path: &Path) -> Result<bool> {
    if !path.is_dir() {
        return Ok(false);
    }
    for entry in fs::read_dir(path).with_context(|| format!("failed to read {}", path.display()))? {
        let entry = entry.with_context(|| format!("failed to read {}", path.display()))?;
        if !is_ignored_entry_name(&entry.file_name()) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn is_ignored_entry_name(name: &OsStr) -> bool {
    let Some(name) = name.to_str() else {
        return false;
    };
    if IGNORED_NAMES
        .iter()
        .any(|ignored| name.eq_ignore_ascii_case(ignored))
    {
        return true;
    }
    false
}

fn next_child_runtime_dir(parent: &Path) -> PathBuf {
    let preferred = parent.join(DEFAULT_CHILD_DIR);
    if !preferred.exists() {
        return preferred;
    }

    for index in 2..1000 {
        let candidate = parent.join(format!("{DEFAULT_CHILD_DIR}-{index}"));
        if !candidate.exists() {
            return candidate;
        }
    }

    parent.join(format!("{DEFAULT_CHILD_DIR}-{}", std::process::id()))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn ignores_only_runtime_noise() {
        let temp = TempDir::new().expect("temp directory");
        fs::write(temp.path().join(".env"), "").expect("write .env");
        fs::write(temp.path().join("haze.exe"), "").expect("write host executable");

        assert!(!has_meaningful_entries(temp.path()).expect("scan directory"));
    }

    #[test]
    fn detects_meaningful_files() {
        let temp = TempDir::new().expect("temp directory");
        fs::write(temp.path().join("notes.txt"), "").expect("write notes");

        assert!(has_meaningful_entries(temp.path()).expect("scan directory"));
    }

    #[test]
    fn marker_disables_prompt() {
        let temp = TempDir::new().expect("temp directory");
        fs::write(temp.path().join("notes.txt"), "").expect("write notes");
        mark_runtime_dir(temp.path()).expect("mark runtime");

        assert!(!should_prompt_for_runtime_dir(temp.path()).expect("prompt check"));
    }

    #[test]
    fn child_runtime_dir_gets_suffix_when_default_exists() {
        let temp = TempDir::new().expect("temp directory");
        fs::create_dir(temp.path().join(DEFAULT_CHILD_DIR)).expect("create child");

        assert_eq!(
            next_child_runtime_dir(temp.path()),
            temp.path().join(format!("{DEFAULT_CHILD_DIR}-2"))
        );
    }

    #[test]
    fn legacy_runtime_prefix_maps_into_separate_runtime_root() {
        let app_dir = Path::new("/opt/haze");
        let runtime_dir = Path::new("/srv/haze");

        assert_eq!(
            resolve_configured_runtime_path(
                app_dir,
                runtime_dir,
                Path::new("runtime/state/goServiceRuntime.json"),
            ),
            runtime_dir.join("state/goServiceRuntime.json")
        );
    }

    #[test]
    fn legacy_runtime_prefix_is_preserved_for_in_place_runtime() {
        let runtime_dir = Path::new("/opt/haze");

        assert_eq!(
            resolve_configured_runtime_path(
                runtime_dir,
                runtime_dir,
                Path::new("runtime/state/goServiceRuntime.json"),
            ),
            runtime_dir.join("runtime/state/goServiceRuntime.json")
        );
    }

    #[test]
    fn seeds_runtime_from_app_bundle() {
        let app = TempDir::new().expect("app directory");
        let runtime = TempDir::new().expect("runtime directory");
        fs::write(app.path().join("config.yaml"), "version: test").expect("write config");
        fs::create_dir(app.path().join("webroot")).expect("create webroot");
        fs::write(app.path().join("webroot").join("index.html"), "").expect("write index");
        fs::create_dir(app.path().join("managed")).expect("create managed");
        fs::write(app.path().join("managed").join("dictionary.json"), "{}")
            .expect("write dictionary");
        fs::write(
            app.path().join("managed").join("receiver_credentials.json"),
            "{}",
        )
        .expect("write credentials");

        prepare_runtime_dir(app.path(), runtime.path()).expect("prepare runtime");

        assert!(runtime.path().join("config.yaml").is_file());
        assert!(runtime.path().join("webroot").join("index.html").is_file());
        assert!(runtime
            .path()
            .join("managed")
            .join("dictionary.json")
            .is_file());
        assert!(!runtime
            .path()
            .join("managed")
            .join("receiver_credentials.json")
            .exists());
        assert!(runtime.path().join("data").join("alerts").is_dir());
    }
}
