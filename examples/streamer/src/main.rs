use clap::Parser;
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;

use kornia_rs::io::fps_counter::FpsCounter;
use kornia_rs::{image::ImageSize, io::stream::StreamCapture};

#[derive(Parser)]
struct Args {
    #[arg(short, long, default_value = "http://192.168.1.156:81/stream")]
    url: String,

    #[arg(short, long)]
    duration: Option<u64>,
}

#[derive(thiserror::Error, Debug)]
pub enum CancelledError {
    #[error("Cancelled")]
    Cancelled,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // start the recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia Streamer App").spawn()?;

    // create a webcam capture object with camera id 0
    // and force the image size to 640x480
    let mut stream = StreamCapture::new(
        "souphttpsrc location=http://192.168.1.156:81/stream ! jpegparse ! jpegdec ! videoconvert ! appsink name=sink",
    )?;

    // create a cancel token to stop the webcam capture
    let cancel_token = CancellationToken::new();
    let child_token = cancel_token.child_token();

    let fps_counter = Arc::new(Mutex::new(FpsCounter::new()));

    ctrlc::set_handler({
        let cancel_token = cancel_token.clone();
        move || {
            println!("Received Ctrl-C signal. Sending cancel signal !!");
            cancel_token.cancel();
        }
    })?;

    let join_handle = tokio::spawn(async move {
        tokio::select! {
            _ = stream.run(|img| {
                    // update the fps counter
                    //fps_counter
                    //    .lock()
                    //    .expect("Failed to lock fps counter")
                    //    .new_frame();

                    // log the image
                    //rec.log_static("image", &rerun::Image::try_from(img.data)?)?;
                    Ok(())
                }) => { Ok(()) }
            _ = child_token.cancelled() => {
                println!("Received cancel signal. Closing webcam.");
                std::mem::drop(stream);
                Err(CancelledError::Cancelled)
            }
        }
    });

    // we launch a timer to cancel the token after a certain duration
    tokio::spawn(async move {
        if let Some(duration_secs) = args.duration {
            tokio::time::sleep(tokio::time::Duration::from_secs(duration_secs)).await;
            println!("Sending cancel signal !!");
            cancel_token.cancel();
        }
    });

    join_handle.await??;

    println!("Finished recording. Closing app.");

    Ok(())
}
