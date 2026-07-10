//! Version-independent FFmpeg runtime discovery.
//!
//! Haze's PCM path does not require FFmpeg. When native FFmpeg support is
//! requested, this module discovers shared libraries at runtime and probes
//! only stable version symbols. No FFmpeg structs cross the ABI boundary, so
//! a new FFmpeg major release cannot make Haze fail to compile.

use std::collections::HashSet;
use std::env;
use std::ffi::{c_char, c_uint, CStr, OsStr};
use std::fs;
use std::path::{Path, PathBuf};

use libloading::Library;

use crate::BackendStatus;

type VersionFn = unsafe extern "C" fn() -> c_uint;
type VersionInfoFn = unsafe extern "C" fn() -> *const c_char;

#[derive(Debug)]
struct ComponentVersion {
    component: &'static str,
    version: c_uint,
}

#[derive(Debug)]
struct ProbeResult {
    version: String,
    partial_fallback: bool,
}

impl ComponentVersion {
    fn display(&self) -> String {
        format!(
            "{} {}.{}.{}",
            self.component,
            self.version >> 16,
            (self.version >> 8) & 0xff,
            self.version & 0xff
        )
    }
}

pub(crate) fn status() -> BackendStatus {
    match probe_ffmpeg() {
        Some(probe) => BackendStatus {
            name: "ffmpeg-runtime",
            available: true,
            version: Some(probe.version),
            fallback: probe.partial_fallback,
        },
        None => BackendStatus {
            name: "builtin-pcm",
            available: true,
            version: Some(
                "native FFmpeg libraries unavailable, using built-in PCM fallback".to_string(),
            ),
            fallback: true,
        },
    }
}

fn probe_ffmpeg() -> Option<ProbeResult> {
    let directories = search_directories();
    let (ffmpeg_version, preferred_directory, avutil) = probe_avutil(&directories)?;
    let mut versions = vec![avutil.display()];
    let mut missing = Vec::new();

    for (component, symbol) in [
        ("avcodec", b"avcodec_version\0".as_slice()),
        ("avformat", b"avformat_version\0".as_slice()),
        ("avfilter", b"avfilter_version\0".as_slice()),
        ("swresample", b"swresample_version\0".as_slice()),
        ("swscale", b"swscale_version\0".as_slice()),
    ] {
        if let Some(version) = probe_component(
            component,
            symbol,
            &directories,
            preferred_directory.as_deref(),
        ) {
            versions.push(version.display());
        } else {
            missing.push(component);
        }
    }

    if !missing.is_empty() {
        versions.push(format!(
            "{} unavailable (built-in fallback active)",
            missing.join("/")
        ));
    }
    Some(ProbeResult {
        version: format!("ffmpeg {ffmpeg_version}, {}", versions.join(", ")),
        partial_fallback: !missing.is_empty(),
    })
}

fn probe_avutil(directories: &[PathBuf]) -> Option<(String, Option<PathBuf>, ComponentVersion)> {
    for candidate in library_candidates("avutil", directories, None) {
        // SAFETY: Loading a user-selected native library is the purpose of
        // this opt-in backend. No data structures are shared with the library.
        let library = match unsafe { load_library(&candidate) } {
            Ok(library) => library,
            Err(_) => continue,
        };

        // SAFETY: These FFmpeg functions have had the same no-argument ABI
        // throughout the supported releases. Symbols are called only while
        // the owning library remains loaded.
        let result = unsafe {
            let version_info = match library.get::<VersionInfoFn>(b"av_version_info\0") {
                Ok(version_info) => version_info,
                Err(_) => continue,
            };
            let version = match library.get::<VersionFn>(b"avutil_version\0") {
                Ok(version) => version,
                Err(_) => continue,
            };
            let version_info = version_info();
            if version_info.is_null() {
                None
            } else {
                Some((
                    CStr::from_ptr(version_info).to_string_lossy().into_owned(),
                    version(),
                ))
            }
        };

        if let Some((version_info, version)) = result {
            return Some((
                version_info,
                candidate.parent().map(Path::to_path_buf),
                ComponentVersion {
                    component: "avutil",
                    version,
                },
            ));
        }
    }
    None
}

fn probe_component(
    component: &'static str,
    symbol: &[u8],
    directories: &[PathBuf],
    preferred_directory: Option<&Path>,
) -> Option<ComponentVersion> {
    for candidate in library_candidates(component, directories, preferred_directory) {
        // SAFETY: See `probe_avutil`. Only a stable version function is used.
        let library = match unsafe { load_library(&candidate) } {
            Ok(library) => library,
            Err(_) => continue,
        };
        // SAFETY: The requested symbols all use FFmpeg's stable version ABI
        // and the symbol is invoked before `library` is dropped.
        let version = unsafe {
            match library.get::<VersionFn>(symbol) {
                Ok(version) => version(),
                Err(_) => continue,
            }
        };
        return Some(ComponentVersion { component, version });
    }
    None
}

unsafe fn load_library(path: &Path) -> Result<Library, libloading::Error> {
    #[cfg(windows)]
    if path.is_absolute() {
        use libloading::os::windows::{Library as WindowsLibrary, LOAD_WITH_ALTERED_SEARCH_PATH};

        // SAFETY: The caller accepts execution of the selected library's
        // initialization routines. This flag also resolves dependencies from
        // the selected FFmpeg directory without changing process-wide PATH.
        return unsafe { WindowsLibrary::load_with_flags(path, LOAD_WITH_ALTERED_SEARCH_PATH) }
            .map(Into::into);
    }

    // SAFETY: The caller accepts execution of the selected library's
    // initialization routines.
    unsafe { Library::new(path) }
}

fn search_directories() -> Vec<PathBuf> {
    let mut directories = Vec::new();
    let mut seen = HashSet::new();

    for variable in ["HAZE_FFMPEG_LIB_DIR", "FFMPEG_DLL_PATH"] {
        if let Some(value) = env::var_os(variable) {
            let path = PathBuf::from(&value);
            if path.is_dir() {
                push_unique(&mut directories, &mut seen, path);
            } else if let Some(parent) = path.parent() {
                push_unique(&mut directories, &mut seen, parent.to_path_buf());
            }
        }
    }

    if let Ok(executable) = env::current_exe() {
        if let Some(parent) = executable.parent() {
            push_unique(&mut directories, &mut seen, parent.to_path_buf());
        }
    }

    for variable in ["PATH", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"] {
        if let Some(value) = env::var_os(variable) {
            for path in env::split_paths(&value) {
                push_unique(&mut directories, &mut seen, path);
            }
        }
    }

    #[cfg(unix)]
    for directory in unix_library_directories() {
        push_unique(&mut directories, &mut seen, directory);
    }

    directories
}

fn push_unique(paths: &mut Vec<PathBuf>, seen: &mut HashSet<PathBuf>, path: PathBuf) {
    if !path.as_os_str().is_empty() && seen.insert(path.clone()) {
        paths.push(path);
    }
}

#[cfg(unix)]
fn unix_library_directories() -> Vec<PathBuf> {
    let mut directories = vec![
        PathBuf::from("/lib"),
        PathBuf::from("/lib64"),
        PathBuf::from("/usr/lib"),
        PathBuf::from("/usr/lib64"),
        PathBuf::from("/usr/local/lib"),
        PathBuf::from("/usr/local/lib64"),
        PathBuf::from("/opt/homebrew/lib"),
    ];
    for root in [Path::new("/lib"), Path::new("/usr/lib")] {
        if let Ok(entries) = fs::read_dir(root) {
            directories.extend(
                entries
                    .filter_map(Result::ok)
                    .map(|entry| entry.path())
                    .filter(|path| path.is_dir()),
            );
        }
    }
    directories
}

fn library_candidates(
    component: &str,
    directories: &[PathBuf],
    preferred_directory: Option<&Path>,
) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    let mut seen = HashSet::new();

    let mut ordered_directories = Vec::with_capacity(directories.len() + 1);
    if let Some(directory) = preferred_directory {
        ordered_directories.push(directory);
    }
    ordered_directories.extend(directories.iter().map(PathBuf::as_path));

    for directory in ordered_directories {
        let first_new_candidate = candidates.len();
        collect_component_libraries(component, directory, &mut candidates, &mut seen);
        candidates[first_new_candidate..].sort_by(|left, right| {
            library_abi_rank(right)
                .cmp(&library_abi_rank(left))
                .then_with(|| left.cmp(right))
        });
    }

    for name in unversioned_library_names(component) {
        let candidate = PathBuf::from(name);
        if seen.insert(candidate.clone()) {
            candidates.push(candidate);
        }
    }
    candidates
}

fn collect_component_libraries(
    component: &str,
    directory: &Path,
    candidates: &mut Vec<PathBuf>,
    seen: &mut HashSet<PathBuf>,
) {
    let Ok(entries) = fs::read_dir(directory) else {
        return;
    };
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if path.is_file()
            && is_component_library(
                component,
                path.file_name().unwrap_or_else(|| OsStr::new("")),
            )
            && seen.insert(path.clone())
        {
            candidates.push(path);
        }
    }
}

fn is_component_library(component: &str, file_name: &OsStr) -> bool {
    let name = file_name.to_string_lossy().to_ascii_lowercase();
    let component = component.to_ascii_lowercase();

    #[cfg(windows)]
    return (name == format!("{component}.dll")
        || name == format!("lib{component}.dll")
        || name.starts_with(&format!("{component}-"))
        || name.starts_with(&format!("lib{component}-")))
        && name.ends_with(".dll");

    #[cfg(target_os = "macos")]
    return name == format!("lib{component}.dylib")
        || (name.starts_with(&format!("lib{component}.")) && name.ends_with(".dylib"));

    #[cfg(all(unix, not(target_os = "macos")))]
    return name == format!("lib{component}.so")
        || name.starts_with(&format!("lib{component}.so."));
}

fn unversioned_library_names(component: &str) -> Vec<String> {
    #[cfg(windows)]
    return vec![format!("{component}.dll"), format!("lib{component}.dll")];

    #[cfg(target_os = "macos")]
    return vec![format!("lib{component}.dylib")];

    #[cfg(all(unix, not(target_os = "macos")))]
    return vec![format!("lib{component}.so")];
}

fn library_abi_rank(path: &Path) -> u32 {
    path.file_name()
        .and_then(OsStr::to_str)
        .map(|name| {
            name.split(|character: char| !character.is_ascii_digit())
                .filter_map(|part| part.parse::<u32>().ok())
                .max()
                .unwrap_or(0)
        })
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_number_is_decoded_without_ffmpeg_headers() {
        let version = ComponentVersion {
            component: "avutil",
            version: (60 << 16) | (26 << 8) | 101,
        };

        assert_eq!(version.display(), "avutil 60.26.101");
    }

    #[test]
    fn abi_rank_accepts_current_and_future_library_names() {
        assert_eq!(library_abi_rank(Path::new("avutil-60.dll")), 60);
        assert_eq!(library_abi_rank(Path::new("libavutil.so.61")), 61);
        assert_eq!(library_abi_rank(Path::new("libavutil.62.dylib")), 62);
    }

    #[test]
    fn component_match_is_not_tied_to_a_known_abi() {
        #[cfg(windows)]
        assert!(is_component_library("avutil", OsStr::new("avutil-99.dll")));
        #[cfg(target_os = "macos")]
        assert!(is_component_library(
            "avutil",
            OsStr::new("libavutil.99.dylib")
        ));
        #[cfg(all(unix, not(target_os = "macos")))]
        assert!(is_component_library(
            "avutil",
            OsStr::new("libavutil.so.99")
        ));
    }
}
