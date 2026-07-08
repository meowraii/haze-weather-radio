use anyhow::Result;
use clap::Parser;

fn main() -> Result<()> {
    let args = haze::DaemonArgs::parse();
    #[cfg(windows)]
    {
        if args.service {
            return haze::run_windows_service(args);
        }
    }
    haze::run(args)
}
