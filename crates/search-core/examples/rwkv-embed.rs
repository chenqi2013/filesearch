#[path = "../src/rwkv.rs"]
mod rwkv;

use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;
use std::io::{BufRead, Write};
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    cpu_threads: Option<usize>,
    #[arg(long)]
    no_spinning: bool,
}

#[derive(Deserialize)]
struct Request {
    tokens: Vec<Vec<i64>>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut model = if args.cpu_threads.is_some() || args.no_spinning {
        rwkv::RwkvModel::load_cpu_config(
            &args.model,
            args.cpu_threads.unwrap_or(4),
            !args.no_spinning,
        )?
    } else {
        rwkv::RwkvModel::load(&args.model)?
    };
    eprintln!("Loaded {}", rwkv::EMBEDDING_PROFILE);
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout().lock();
    for line in stdin.lock().lines() {
        let request: Request = serde_json::from_str(&line?).context("Invalid embedding request")?;
        let vectors = model.embed_tokens(&request.tokens)?;
        serde_json::to_writer(&mut stdout, &vectors)?;
        writeln!(stdout)?;
        stdout.flush()?;
    }
    Ok(())
}
