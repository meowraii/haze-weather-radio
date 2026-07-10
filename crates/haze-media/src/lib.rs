//! Shared media primitives for Haze.
//!
//! This crate is the boundary between Haze's event-driven audio bus and the
//! codec/container backends used to decode, normalize, and encode media.

mod pcm;

#[cfg(feature = "ffmpeg-runtime")]
mod ffmpeg_runtime;

pub use pcm::{
    decode_wav, normalize_pcm, pcm16_samples, push_i16, read_i16, remix_pcm16, resample_pcm16,
    silence_chunk, AudioFormat, Pcm,
};

/// Reports which media backend is compiled into this build.
#[must_use]
pub fn backend_status() -> BackendStatus {
    backend::status()
}

/// Runtime information for the compiled media backend.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct BackendStatus {
    pub name: &'static str,
    pub available: bool,
    pub version: Option<String>,
    /// True when Haze is using its built-in PCM implementation for all or part
    /// of the media path because an optional native component could not load.
    pub fallback: bool,
}

#[cfg(feature = "ffmpeg-runtime")]
mod backend {
    use super::BackendStatus;

    pub(crate) fn status() -> BackendStatus {
        super::ffmpeg_runtime::status()
    }
}

#[cfg(not(feature = "ffmpeg-runtime"))]
mod backend {
    use super::BackendStatus;

    pub(crate) fn status() -> BackendStatus {
        BackendStatus {
            name: "builtin-pcm",
            available: true,
            version: None,
            fallback: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_backend_is_available() {
        let status = backend_status();
        assert!(status.available);
        assert!(!status.name.is_empty());
    }
}
