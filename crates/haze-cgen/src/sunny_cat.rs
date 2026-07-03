include!(concat!(env!("OUT_DIR"), "/sunny_cat_asset.rs"));

pub(crate) fn available() -> bool {
    AVAILABLE && RGBA.len() == WIDTH.saturating_mul(HEIGHT).saturating_mul(4)
}
