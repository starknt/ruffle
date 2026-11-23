use std::{env, fs, io, path::PathBuf};

fn main() -> io::Result<()> {
    println!("cargo:rerun-if-changed=asc.jar");
    // Cached build scripts and test runners may outlive the source worktree.
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo supplies OUT_DIR"));
    fs::copy("asc.jar", out_dir.join("asc.jar"))?;
    Ok(())
}
