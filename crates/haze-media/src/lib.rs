//! Shared media primitives for Haze.
//!
//! This crate is the boundary between Haze's event-driven audio bus and the
//! codec/container backends used to decode, normalize, and encode media.

mod pcm;

pub use pcm::{
    decode_wav, normalize_pcm, pcm16_samples, push_i16, read_i16, remix_pcm16, resample_pcm16,
    silence_chunk, AudioFormat, Pcm,
};

/// Reports which media backend is compiled into this build.
#[must_use]
pub fn backend_status() -> BackendStatus {
    BackendStatus {
        name: backend::name(),
        available: backend::available(),
        version: backend::version(),
    }
}

/// Runtime information for the compiled media backend.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct BackendStatus {
    pub name: &'static str,
    pub available: bool,
    pub version: Option<String>,
}

#[cfg(feature = "ffmpeg-rsmpeg")]
mod backend {
    pub(crate) fn name() -> &'static str {
        "rsmpeg/libav"
    }

    pub(crate) fn available() -> bool {
        true
    }

    pub(crate) fn version() -> Option<String> {
        let ffmpeg = rsmpeg::avutil::version_info().to_string_lossy();
        Some(format!(
            "ffmpeg {ffmpeg}, avcodec {}, avformat {}, avutil {}",
            rsmpeg::avcodec::version(),
            rsmpeg::avformat::version(),
            rsmpeg::avutil::version()
        ))
    }
}

#[cfg(not(feature = "ffmpeg-rsmpeg"))]
mod backend {
    pub(crate) fn name() -> &'static str {
        "builtin-pcm"
    }

    pub(crate) fn available() -> bool {
        true
    }

    pub(crate) fn version() -> Option<String> {
        None
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
