use anyhow::{Ok, Result};
use clap::Parser;
use indicatif::{ParallelProgressIterator, ProgressStyle};
use kornia_rs::resize;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rayon::ThreadPoolBuilder;
use std::env::var_os;
use std::path::PathBuf;
use std::time::Instant;
use walkdir::WalkDir;

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long)]
    images_dir: PathBuf,

    #[arg(short, long)]
    output_dir: PathBuf,

    #[arg(short, long)]
    size: usize,

    #[arg(short, long, default_value = "8")]
    num_threads: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("{:?}", args);

    ThreadPoolBuilder::new()
        .num_threads(args.num_threads)
        .build_global()
        .expect("Failed to build thread pool");

    // Walk through the images directory and collect all the jpeg images
    let images_paths: Vec<PathBuf> = WalkDir::new(&args.images_dir)
        .into_iter()
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry.file_type().is_file()
                && entry
                    .path()
                    .extension()
                    .map(|ext| ext == "jpeg")
                    .unwrap_or(false)
        })
        .map(|entry| entry.path().to_path_buf())
        .collect();

    if images_paths.is_empty() {
        println!("No images found in the directory");
        return Ok(());
    }

    // create the output directory if it doesn't exist
    if !args.output_dir.exists() {
        std::fs::create_dir_all(&args.output_dir)?;
    }

    // Create a progress bar
    let pb = indicatif::ProgressBar::new(images_paths.len() as u64);
    pb.set_style(ProgressStyle::default_bar().template(
        "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} ({eta}) {msg} {per_sec} img/sec",
    )?.progress_chars("##>-"));

    let mut total_time = vec![];

    for _ in 0..10 {
        let t0 = Instant::now();

        // load the images in parallel
        images_paths
            .par_iter()
            .progress_with(pb.clone())
            .for_each(|file_path| {
                // read the image
                let img = kornia_rs::io::functional::read_image_jpeg(file_path).unwrap();

                // resize the image
                //let resized_img = kornia_rs::resize::resize_native(
                //    &img.cast::<f32>().unwrap(),
                //    kornia_rs::image::ImageSize {
                //        width: args.size,
                //        height: args.size,
                //    },
                //    kornia_rs::resize::InterpolationMode::Bilinear,
                //)
                //.unwrap()
                //.cast::<u8>()
                //.unwrap();

                let resized_img = kornia_rs::resize::resize_fast(
                    &img,
                    kornia_rs::image::ImageSize {
                        width: args.size,
                        height: args.size,
                    },
                    kornia_rs::resize::InterpolationMode::Bilinear,
                )
                .unwrap();

                // save the resized image
                //kornia_rs::io::functional::write_image_jpeg(
                //    &args.output_dir.join(file_path.file_name().unwrap()),
                //    &resized_img,
                //)
                //.unwrap();
            });

        let t1 = Instant::now();
        let dur = t1.duration_since(t0).as_millis();
        total_time.push(dur);

        //println!("Time taken: {:?} ms", t1.duration_since(t0).as_millis());
    }

    let avg_time = total_time.iter().sum::<u128>() / total_time.len() as u128;
    println!("Average time taken: {:?} ms", avg_time);

    Ok(())
}
