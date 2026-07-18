use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::architecture::FeedId;

const CAPS_SCHEMA_VERSION: u16 = 1;
const MAX_CAPS_STATE_BYTES: u64 = 64 * 1024;
const MAX_VIDEO_DIMENSION: u32 = 32_768;
const MAX_RATIONAL_COMPONENT: u32 = 1_000_000;
const MAX_COLORIMETRY_LEN: usize = 64;
const CAPS_WRITER_QUEUE_DEPTH: usize = 8;

static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Error)]
pub(crate) enum SourceCapsError {
    #[error("video width and height must be between 1 and {MAX_VIDEO_DIMENSION}")]
    InvalidDimensions,
    #[error("rational components must be between 1 and {MAX_RATIONAL_COMPONENT}")]
    InvalidRational,
    #[error("progressive video cannot declare an interlaced field order")]
    ProgressiveFieldOrder,
    #[error("interlaced video must declare a field order or unknown")]
    MissingInterlacedFieldOrder,
    #[error("colorimetry is empty, too long, or contains unsupported characters")]
    InvalidColorimetry,
    #[error("invalid caps state path: {0}")]
    InvalidStatePath(String),
    #[error("caps state exceeds {MAX_CAPS_STATE_BYTES} bytes")]
    StateFileTooLarge,
    #[error("unsupported caps state schema version {0}")]
    UnsupportedSchema(u16),
    #[error("failed to access caps state: {0}")]
    Io(#[from] io::Error),
    #[error("failed to decode caps state: {0}")]
    Decode(#[from] serde_json::Error),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CapsRational {
    numerator: NonZeroU32,
    denominator: NonZeroU32,
}

impl CapsRational {
    pub(crate) fn new(numerator: u32, denominator: u32) -> Result<Self, SourceCapsError> {
        if numerator == 0
            || denominator == 0
            || numerator > MAX_RATIONAL_COMPONENT
            || denominator > MAX_RATIONAL_COMPONENT
        {
            return Err(SourceCapsError::InvalidRational);
        }
        let divisor = greatest_common_divisor(numerator, denominator);
        let numerator = NonZeroU32::new(numerator / divisor)
            .expect("a non-zero numerator remains non-zero after reduction");
        let denominator = NonZeroU32::new(denominator / divisor)
            .expect("a non-zero denominator remains non-zero after reduction");
        Ok(Self {
            numerator,
            denominator,
        })
    }

    pub(crate) const fn numerator(self) -> u32 {
        self.numerator.get()
    }

    pub(crate) const fn denominator(self) -> u32 {
        self.denominator.get()
    }
}

fn greatest_common_divisor(mut left: u32, mut right: u32) -> u32 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SourceScanMode {
    Progressive,
    Interlaced,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SourceFieldOrder {
    NotApplicable,
    TopFieldFirst,
    BottomFieldFirst,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SourceCaps {
    width: NonZeroU32,
    height: NonZeroU32,
    frame_rate: CapsRational,
    scan_mode: SourceScanMode,
    field_order: SourceFieldOrder,
    pixel_aspect_ratio: CapsRational,
    colorimetry: String,
}

impl SourceCaps {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        width: u32,
        height: u32,
        frame_rate_numerator: u32,
        frame_rate_denominator: u32,
        scan_mode: SourceScanMode,
        field_order: SourceFieldOrder,
        pixel_aspect_ratio_numerator: u32,
        pixel_aspect_ratio_denominator: u32,
        colorimetry: impl AsRef<str>,
    ) -> Result<Self, SourceCapsError> {
        let width = NonZeroU32::new(width).ok_or(SourceCapsError::InvalidDimensions)?;
        let height = NonZeroU32::new(height).ok_or(SourceCapsError::InvalidDimensions)?;
        if width.get() > MAX_VIDEO_DIMENSION || height.get() > MAX_VIDEO_DIMENSION {
            return Err(SourceCapsError::InvalidDimensions);
        }
        match (scan_mode, field_order) {
            (SourceScanMode::Progressive, SourceFieldOrder::NotApplicable) => {}
            (SourceScanMode::Progressive, _) => {
                return Err(SourceCapsError::ProgressiveFieldOrder);
            }
            (SourceScanMode::Interlaced, SourceFieldOrder::NotApplicable) => {
                return Err(SourceCapsError::MissingInterlacedFieldOrder);
            }
            (SourceScanMode::Interlaced, _) => {}
        }
        let colorimetry = normalize_colorimetry(colorimetry.as_ref())?;
        Ok(Self {
            width,
            height,
            frame_rate: CapsRational::new(frame_rate_numerator, frame_rate_denominator)?,
            scan_mode,
            field_order,
            pixel_aspect_ratio: CapsRational::new(
                pixel_aspect_ratio_numerator,
                pixel_aspect_ratio_denominator,
            )?,
            colorimetry,
        })
    }

    pub(crate) fn fallback() -> Self {
        Self::new(
            720,
            480,
            30_000,
            1_001,
            SourceScanMode::Progressive,
            SourceFieldOrder::NotApplicable,
            1,
            1,
            "unknown",
        )
        .expect("built-in source caps are valid")
    }

    pub(crate) const fn width(&self) -> u32 {
        self.width.get()
    }

    pub(crate) const fn height(&self) -> u32 {
        self.height.get()
    }

    pub(crate) const fn frame_rate(&self) -> CapsRational {
        self.frame_rate
    }

    pub(crate) const fn scan_mode(&self) -> SourceScanMode {
        self.scan_mode
    }

    pub(crate) const fn field_order(&self) -> SourceFieldOrder {
        self.field_order
    }

    pub(crate) const fn pixel_aspect_ratio(&self) -> CapsRational {
        self.pixel_aspect_ratio
    }

    pub(crate) fn colorimetry(&self) -> &str {
        &self.colorimetry
    }

    fn validate(&self) -> Result<(), SourceCapsError> {
        if normalize_colorimetry(&self.colorimetry)? != self.colorimetry {
            return Err(SourceCapsError::InvalidColorimetry);
        }
        let canonical = Self::new(
            self.width(),
            self.height(),
            self.frame_rate.numerator(),
            self.frame_rate.denominator(),
            self.scan_mode,
            self.field_order,
            self.pixel_aspect_ratio.numerator(),
            self.pixel_aspect_ratio.denominator(),
            &self.colorimetry,
        )?;
        if canonical != *self {
            return Err(SourceCapsError::InvalidRational);
        }
        Ok(())
    }
}

fn normalize_colorimetry(raw: &str) -> Result<String, SourceCapsError> {
    let value = raw.trim().to_ascii_lowercase();
    if value.is_empty()
        || value.len() > MAX_COLORIMETRY_LEN
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b':' | b'-' | b'_' | b'.'))
    {
        return Err(SourceCapsError::InvalidColorimetry);
    }
    Ok(value)
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CapsStateFile {
    schema_version: u16,
    caps: SourceCaps,
}

#[derive(Debug, Clone)]
pub(crate) struct LastKnownCapsStore {
    path: PathBuf,
}

impl LastKnownCapsStore {
    pub(crate) fn new(base_dir: &Path, feed_id: &FeedId) -> Result<Self, SourceCapsError> {
        let canonical_base = fs::canonicalize(base_dir)?;
        let state_dir = create_contained_state_directory(&canonical_base)?;
        let canonical_state_dir = fs::canonicalize(&state_dir)?;
        if !canonical_state_dir.starts_with(&canonical_base) {
            return Err(SourceCapsError::InvalidStatePath(
                "caps directory escapes the configured base directory".to_string(),
            ));
        }

        Ok(Self {
            path: state_dir.join(format!("{}.json", feed_id.as_str())),
        })
    }

    #[cfg(test)]
    fn from_path(path: PathBuf) -> Self {
        Self { path }
    }

    pub(crate) fn load(&self) -> Result<Option<SourceCaps>, SourceCapsError> {
        let metadata = match fs::symlink_metadata(&self.path) {
            Ok(metadata) => metadata,
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(err) => return Err(err.into()),
        };
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(SourceCapsError::InvalidStatePath(
                "caps state is not a regular file".to_string(),
            ));
        }
        if metadata.len() > MAX_CAPS_STATE_BYTES {
            return Err(SourceCapsError::StateFileTooLarge);
        }

        let mut file = File::open(&self.path)?;
        let mut raw = Vec::with_capacity(usize::try_from(metadata.len()).unwrap_or(0));
        Read::by_ref(&mut file)
            .take(MAX_CAPS_STATE_BYTES + 1)
            .read_to_end(&mut raw)?;
        if raw.len() as u64 > MAX_CAPS_STATE_BYTES {
            return Err(SourceCapsError::StateFileTooLarge);
        }
        let state: CapsStateFile = serde_json::from_slice(&raw)?;
        if state.schema_version != CAPS_SCHEMA_VERSION {
            return Err(SourceCapsError::UnsupportedSchema(state.schema_version));
        }
        state.caps.validate()?;
        Ok(Some(state.caps))
    }

    pub(crate) fn save(&self, caps: &SourceCaps) -> Result<(), SourceCapsError> {
        caps.validate()?;
        if let Ok(metadata) = fs::symlink_metadata(&self.path) {
            if metadata.file_type().is_symlink() || !metadata.is_file() {
                return Err(SourceCapsError::InvalidStatePath(
                    "caps state is not a regular file".to_string(),
                ));
            }
        }
        let state = CapsStateFile {
            schema_version: CAPS_SCHEMA_VERSION,
            caps: caps.clone(),
        };
        let raw = serde_json::to_vec_pretty(&state)?;
        if raw.len() as u64 > MAX_CAPS_STATE_BYTES {
            return Err(SourceCapsError::StateFileTooLarge);
        }

        let temporary = temporary_path(&self.path);
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)?;
        let result = (|| -> Result<(), SourceCapsError> {
            file.write_all(&raw)?;
            file.write_all(b"\n")?;
            file.sync_all()?;
            drop(file);
            atomic_replace(&temporary, &self.path)?;
            sync_parent_directory(&self.path)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        result
    }

    #[cfg(test)]
    fn path(&self) -> &Path {
        &self.path
    }
}

fn create_contained_state_directory(base_dir: &Path) -> Result<PathBuf, SourceCapsError> {
    let mut current = base_dir.to_path_buf();
    for component in ["runtime", "cgen", "caps"] {
        current.push(component);
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
                return Err(SourceCapsError::InvalidStatePath(format!(
                    "{component} is not a regular directory"
                )));
            }
            Ok(_) => {}
            Err(err) if err.kind() == io::ErrorKind::NotFound => match fs::create_dir(&current) {
                Ok(()) => {}
                Err(err) if err.kind() == io::ErrorKind::AlreadyExists => {
                    reject_symlink(&current)?;
                }
                Err(err) => return Err(err.into()),
            },
            Err(err) => return Err(err.into()),
        }
    }
    Ok(current)
}

#[derive(Debug, Clone, Default)]
pub(crate) struct CapsWriterStatus {
    pub(crate) queued: u64,
    pub(crate) dropped: u64,
    pub(crate) written: u64,
    pub(crate) last_written: Option<SourceCaps>,
    pub(crate) last_error: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct CapsWriter {
    sender: mpsc::SyncSender<SourceCaps>,
    status: Arc<Mutex<CapsWriterStatus>>,
    last_enqueued: Arc<Mutex<Option<SourceCaps>>>,
}

impl CapsWriter {
    pub(crate) fn spawn(store: LastKnownCapsStore) -> Result<Self, SourceCapsError> {
        let (sender, receiver) = mpsc::sync_channel::<SourceCaps>(CAPS_WRITER_QUEUE_DEPTH);
        let status = Arc::new(Mutex::new(CapsWriterStatus::default()));
        let writer_status = Arc::clone(&status);
        let last_enqueued = Arc::new(Mutex::new(None));
        let writer_last_enqueued = Arc::clone(&last_enqueued);
        std::thread::Builder::new()
            .name("haze-cgen-caps-writer".to_string())
            .spawn(move || {
                while let Ok(caps) = receiver.recv() {
                    match store.save(&caps) {
                        Ok(()) => {
                            if let Ok(mut status) = writer_status.lock() {
                                status.written = status.written.saturating_add(1);
                                status.last_written = Some(caps.clone());
                                status.last_error = None;
                            }
                        }
                        Err(err) => {
                            if let Ok(mut status) = writer_status.lock() {
                                status.last_error = Some(err.to_string());
                            }
                            if let Ok(mut last_enqueued) = writer_last_enqueued.lock() {
                                *last_enqueued = None;
                            }
                        }
                    }
                }
            })?;
        Ok(Self {
            sender,
            status,
            last_enqueued,
        })
    }

    pub(crate) fn enqueue(&self, caps: &SourceCaps) {
        let mut last_enqueued = match self.last_enqueued.lock() {
            Ok(last_enqueued) => last_enqueued,
            Err(_) => return,
        };
        if last_enqueued.as_ref() == Some(caps) {
            return;
        }
        match self.sender.try_send(caps.clone()) {
            Ok(()) => {
                *last_enqueued = Some(caps.clone());
                if let Ok(mut status) = self.status.lock() {
                    status.queued = status.queued.saturating_add(1);
                }
            }
            Err(mpsc::TrySendError::Full(_)) => {
                if let Ok(mut status) = self.status.lock() {
                    status.dropped = status.dropped.saturating_add(1);
                }
            }
            Err(mpsc::TrySendError::Disconnected(_)) => {
                if let Ok(mut status) = self.status.lock() {
                    status.last_error = Some("caps persistence worker stopped".to_string());
                }
            }
        }
    }

    pub(crate) fn status(&self) -> CapsWriterStatus {
        self.status
            .lock()
            .map(|status| status.clone())
            .unwrap_or(CapsWriterStatus {
                last_error: Some("caps persistence status lock poisoned".to_string()),
                ..CapsWriterStatus::default()
            })
    }
}

fn reject_symlink(path: &Path) -> Result<(), SourceCapsError> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(SourceCapsError::InvalidStatePath(
            "caps directory is not a regular directory".to_string(),
        ));
    }
    Ok(())
}

fn temporary_path(destination: &Path) -> PathBuf {
    let sequence = TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let file_name = destination
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("caps.json");
    destination.with_file_name(format!(
        ".{file_name}.{}.{}.tmp",
        std::process::id(),
        sequence
    ))
}

#[cfg(not(windows))]
fn atomic_replace(source: &Path, destination: &Path) -> io::Result<()> {
    fs::rename(source, destination)
}

#[cfg(windows)]
fn atomic_replace(source: &Path, destination: &Path) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;

    const MOVEFILE_REPLACE_EXISTING: u32 = 0x0000_0001;
    const MOVEFILE_WRITE_THROUGH: u32 = 0x0000_0008;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn MoveFileExW(existing: *const u16, replacement: *const u16, flags: u32) -> i32;
    }

    let source = source
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let destination = destination
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let replaced = unsafe {
        MoveFileExW(
            source.as_ptr(),
            destination.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if replaced == 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(unix)]
fn sync_parent_directory(path: &Path) -> io::Result<()> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };
    File::open(parent)?.sync_all()
}

#[cfg(not(unix))]
fn sync_parent_directory(_path: &Path) -> io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::{
        CapsRational, LastKnownCapsStore, SourceCaps, SourceCapsError, SourceFieldOrder,
        SourceScanMode, MAX_CAPS_STATE_BYTES,
    };
    use crate::architecture::FeedId;

    #[test]
    fn fallback_is_broadcast_safe_ntsc_sd() {
        let caps = SourceCaps::fallback();

        assert_eq!(caps.width(), 720);
        assert_eq!(caps.height(), 480);
        assert_eq!(caps.frame_rate(), CapsRational::new(30_000, 1_001).unwrap());
        assert_eq!(caps.scan_mode(), SourceScanMode::Progressive);
        assert_eq!(caps.field_order(), SourceFieldOrder::NotApplicable);
        assert_eq!(caps.pixel_aspect_ratio(), CapsRational::new(1, 1).unwrap());
    }

    #[test]
    fn validates_scan_and_field_order_together() {
        let error = SourceCaps::new(
            1920,
            1080,
            30_000,
            1_001,
            SourceScanMode::Progressive,
            SourceFieldOrder::TopFieldFirst,
            1,
            1,
            "bt709",
        )
        .expect_err("progressive field order must be rejected");

        assert!(matches!(error, SourceCapsError::ProgressiveFieldOrder));
    }

    #[test]
    fn rational_values_are_reduced_for_stable_caps_comparison() {
        assert_eq!(
            CapsRational::new(60_000, 2_002).unwrap(),
            CapsRational::new(30_000, 1_001).unwrap()
        );
    }

    #[test]
    fn store_round_trips_and_replaces_atomically() {
        let directory = tempdir().unwrap();
        let feed_id = FeedId::parse("CFSP-CGEN").unwrap();
        let store = LastKnownCapsStore::new(directory.path(), &feed_id).unwrap();
        let first = SourceCaps::fallback();
        let second = SourceCaps::new(
            1920,
            1080,
            60_000,
            1_001,
            SourceScanMode::Progressive,
            SourceFieldOrder::NotApplicable,
            1,
            1,
            "bt709",
        )
        .unwrap();

        assert!(store.load().unwrap().is_none());
        store.save(&first).unwrap();
        assert_eq!(store.load().unwrap(), Some(first));
        store.save(&second).unwrap();
        assert_eq!(store.load().unwrap(), Some(second));
        let canonical_base = fs::canonicalize(directory.path()).unwrap();
        assert!(store
            .path()
            .starts_with(canonical_base.join("runtime").join("cgen").join("caps")));
    }

    #[test]
    fn rejects_oversized_state_before_decoding() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("caps.json");
        fs::write(&path, vec![b' '; MAX_CAPS_STATE_BYTES as usize + 1]).unwrap();
        let store = LastKnownCapsStore::from_path(path);

        assert!(matches!(
            store.load(),
            Err(SourceCapsError::StateFileTooLarge)
        ));
    }

    #[test]
    fn rejects_unknown_fields_and_invalid_persisted_caps() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("caps.json");
        fs::write(
            &path,
            br#"{
                "schema_version": 1,
                "caps": {
                    "width": 1920,
                    "height": 1080,
                    "frame_rate": {"numerator": 30000, "denominator": 1001},
                    "scan_mode": "progressive",
                    "field_order": "not_applicable",
                    "pixel_aspect_ratio": {"numerator": 1, "denominator": 1},
                    "colorimetry": "bt709",
                    "unexpected": true
                }
            }"#,
        )
        .unwrap();
        let store = LastKnownCapsStore::from_path(path);

        assert!(matches!(store.load(), Err(SourceCapsError::Decode(_))));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlinked_state_directory() {
        use std::os::unix::fs::symlink;

        let directory = tempdir().unwrap();
        let outside = tempdir().unwrap();
        fs::create_dir_all(directory.path().join("runtime").join("cgen")).unwrap();
        symlink(
            outside.path(),
            directory.path().join("runtime").join("cgen").join("caps"),
        )
        .unwrap();
        let feed_id = FeedId::parse("feed").unwrap();

        assert!(matches!(
            LastKnownCapsStore::new(directory.path(), &feed_id),
            Err(SourceCapsError::InvalidStatePath(_))
        ));
    }
}
