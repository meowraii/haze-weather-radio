use anyhow::Result;
use clap::Parser;

fn main() -> Result<()> {
    haze::run(haze::DaemonArgs::parse())
}
