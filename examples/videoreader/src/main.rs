use clap::Parser;
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};
use tokio_util::sync::CancellationToken;

use kornia_rs::io::video::{self, VideoReader, VideoWriter};
use kornia_rs::{image::Image, io::stream::StreamCapture};

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    video_file: PathBuf,

    #[arg(short, long, default_value = "1.0")]
    scale_factor: f32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // start the recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia Video App").spawn()?;

    //let pipeline_str = format!(
    //    "filesrc location={} ! decodebin ! videoconvert ! video/x-raw,format=RGB ! appsink name=sink",
    //    args.video_file.to_str().unwrap()
    //);
    //let mut video_capture = StreamCapture::new(&pipeline_str)?;

    //video_capture
    //    .run(|img| {
    //        // compute the new image size
    //        let new_size = kornia_rs::image::ImageSize {
    //            width: (img.size().width as f32 * args.scale_factor) as usize,
    //            height: (img.size().height as f32 * args.scale_factor) as usize,
    //        };

    //        // resize the image
    //        let img = kornia_rs::resize::resize_fast(
    //            &img,
    //            new_size,
    //            kornia_rs::resize::InterpolationMode::Bilinear,
    //        )?;

    //        // log the image to the recording stream
    //        rec.log("image", &rerun::Image::try_from(img.data)?)?;

    //        Ok(())
    //    })
    //    .await?;

    let mut video_reader = VideoReader::new(&args.video_file)?;
    let mut video_writer = VideoWriter::new(Path::new(&"output.mp4".to_string()), 30.0, 128, 128)?;

    video_reader.start()?;
    video_writer.start()?;

    let mut counter = 0;

    while let Some(img) = video_reader.grab_frame()? {
        let img_resize = kornia_rs::resize::resize_fast(
            &img,
            kornia_rs::image::ImageSize {
                width: 128,
                height: 128,
            },
            kornia_rs::resize::InterpolationMode::Bilinear,
        )?;
        println!("Image: #{} {}", counter, img.size());
        counter += 1;

        video_writer.write(img_resize)?;
        rec.log("image", &rerun::Image::try_from(img.data)?)?;
    }

    video_reader.stop()?;
    video_writer.stop()?;

    Ok(())
}
