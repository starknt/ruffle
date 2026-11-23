#![recursion_limit = "256"]
pub mod cli;
mod exporter;
mod player_ext;
mod progress;

use anyhow::{Context, Result, anyhow, bail};
use indicatif::ProgressBar;
use rayon::prelude::*;
use std::collections::HashSet;
use std::fs::{create_dir_all, rename};
use std::io::{self, Write};
use std::path::Path;
use walkdir::{DirEntry, WalkDir};

use crate::cli::{FrameSelection, Opt};
use crate::exporter::{Exporter, MovieExport};
use crate::progress::ExporterProgress;

enum Output<'a> {
    File(&'a Path),
    Directory(&'a Path),
    Stdout,
}

fn capture_frames(
    movie: &mut MovieExport,
    swf: &Path,
    output: Output<'_>,
    progress: &ExporterProgress,
) -> Result<()> {
    let count = movie.frames.end - movie.frames.start;
    let digits = count.to_string().len();
    // Keep each PNG on disk until the movie has rendered successfully. Only
    // one captured frame per movie needs to be resident at a time.
    let staging = if let Output::Directory(path) = output {
        create_dir_all(path)?;
        Some(tempfile::tempdir_in(path)?)
    } else {
        None
    };

    for _ in 0..movie.frames.start {
        movie.run_frame();
    }
    for (index, frame) in movie.frames.clone().enumerate() {
        progress.set_message(format!("{} frame {frame}", swf.display()));
        movie.run_frame();
        let image = movie
            .capture_frame()
            .with_context(|| format!("Unable to capture frame {frame} of {}", swf.display()))?;
        match output {
            Output::File(path) => image.save(path)?,
            Output::Stdout => {
                let mut png = io::Cursor::new(Vec::new());
                image.write_to(&mut png, image::ImageFormat::Png)?;
                io::stdout().lock().write_all(png.get_ref())?;
            }
            Output::Directory(_) => {
                let directory = staging.as_ref().expect("frame staging directory");
                image.save(directory.path().join(format!("{index:0digits$}.png")))?;
            }
        }
        progress.inc(1);
    }
    if let (Output::Directory(output), Some(directory)) = (output, staging) {
        for index in 0..count {
            let name = format!("{index:0digits$}.png");
            rename(directory.path().join(&name), output.join(name))?;
        }
    }
    Ok(())
}

fn find_files(root: &Path, with_progress: bool) -> Result<Vec<DirEntry>> {
    let progress = with_progress.then(ProgressBar::new_spinner);
    let mut results = Vec::new();
    let mut destinations = HashSet::new();

    for entry in WalkDir::new(root).follow_links(true) {
        let entry = entry?;
        if entry.file_type().is_file()
            && entry
                .path()
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("swf"))
        {
            if !destinations.insert(entry.path().with_extension("")) {
                bail!(
                    "Multiple SWFs have the same output name at {}",
                    entry.path().display()
                );
            }
            results.push(entry);
            if let Some(progress) = &progress {
                progress.set_message(format!("Searching for swf files... {}", results.len()));
            }
        }
    }
    if let Some(progress) = progress {
        progress.finish_with_message(format!("Found {} swf files to export", results.len()));
    }
    Ok(results)
}

fn capture_single_swf(exporter: &Exporter, opt: &Opt) -> Result<()> {
    let single = opt.frames.is_single_frame();
    let output = opt.output_path.clone().unwrap_or_else(|| {
        Path::new(opt.swf.file_name().expect("SWF file name")).with_extension(if single {
            "png"
        } else {
            ""
        })
    });
    let destination = if single && output == Path::new("-") {
        Output::Stdout
    } else if single {
        Output::File(&output)
    } else {
        Output::Directory(&output)
    };

    let progress = ExporterProgress::new(opt, 1);
    let mut movie = exporter.start_exporting_movie(&opt.swf)?;
    capture_frames(&mut movie, &opt.swf, destination, &progress)?;
    let count = movie.frames.end - movie.frames.start;
    if count == 1 && opt.silent {
        return Ok(());
    }
    let frames = if count == 1 {
        "first frame".to_string()
    } else {
        format!("first {count} frames")
    };
    progress.finish_with_message(format!(
        "Saved {frames} of {} to {}",
        opt.swf.display(),
        output.display()
    ));
    Ok(())
}

fn capture_multiple_swfs(exporter: &Exporter, opt: &Opt, output: &Path) -> Result<()> {
    let files = find_files(&opt.swf, !opt.silent)?;
    let progress = ExporterProgress::new(opt, files.len() as u64);

    let export_file = |entry: &DirEntry| -> Result<()> {
        let file = entry.path();
        let mut movie = exporter.start_exporting_movie(file)?;
        let single = movie.frames.end - movie.frames.start == 1;
        let relative = file.strip_prefix(&opt.swf)?;
        let destination = output.join(relative.with_extension(if single { "png" } else { "" }));
        if single {
            if let Some(parent) = destination.parent() {
                create_dir_all(parent)?;
            }
            capture_frames(&mut movie, file, Output::File(&destination), &progress)
        } else {
            capture_frames(&mut movie, file, Output::Directory(&destination), &progress)
        }
    };
    // A bad input should not suppress valid siblings or turn a partial export into success.
    let failures: Vec<_> = files
        .par_iter()
        .filter_map(|entry| {
            export_file(entry)
                .err()
                .map(|e| format!("{}: {e:#}", entry.path().display()))
        })
        .collect();
    if !failures.is_empty() {
        bail!(
            "Failed to export {} of {} files:\n{}",
            failures.len(),
            files.len(),
            failures.join("\n")
        );
    }
    let frames = match opt.frames {
        FrameSelection::Count(n) if n.get() == 1 => "first frame".to_owned(),
        FrameSelection::Count(n) => format!("first {n} frames"),
        FrameSelection::All => "all frames".to_owned(),
    };
    progress.finish_with_message(format!(
        "Saved {frames} of {} files to {}",
        files.len(),
        output.display()
    ));
    Ok(())
}

pub fn run_main(opt: Opt) -> Result<()> {
    if opt.swf.is_file() {
        capture_single_swf(&Exporter::new(&opt)?, &opt)
    } else if opt.swf.is_dir() {
        let output = opt.output_path.as_deref().ok_or_else(|| {
            anyhow!("Output directory is required when exporting multiple files.")
        })?;
        capture_multiple_swfs(&Exporter::new(&opt)?, &opt, output)
    } else {
        bail!("Not a file or directory: {}", opt.swf.display())
    }
}
