use std::{
    path::{Path, PathBuf},
    sync::atomic::AtomicUsize,
};

use clap::Parser;
use kornia_rs::io::functional as F;
use ndarray::parallel::prelude::{IntoParallelIterator, ParallelIterator};
use polars::prelude::*;
use tokio::sync::Mutex;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The path to the data directory
    #[arg(short, long)]
    data_dir: PathBuf,

    /// The batch size
    #[arg(short, long, default_value = "8")]
    batch_size: usize,

    /// The number of epochs
    #[arg(short, long, default_value = "10")]
    epochs: usize,
}

/// Load the data from the CSV file
///
/// The CSV file should have the following columns:
/// - image_files: The path to the image files
/// - angular_velocity: The angular velocity of the robot
///
/// # Arguments
///
/// * `data_file` - The path to the CSV file
///
/// # Returns
///
/// A vector of tuples containing the image file path and the angular velocity
///
fn load_data(data_file: &Path) -> Result<Vec<(String, f64)>, Box<dyn std::error::Error>> {
    // Load the data from the CSV file
    let df = CsvReader::from_path(data_file)?
        .infer_schema(None)
        .has_header(true)
        .finish()?;

    // Get the columns from the dataframe
    let series = df.columns(["image_files", "angular_velocity"])?;

    // Convert the columns to a vector of tuples
    let data = series[0]
        .str()?
        .into_iter()
        .zip(series[1].f64()?.into_iter())
        .map(|(img, ang_vel)| {
            (
                img.unwrap_or_default().to_string(),
                ang_vel.unwrap_or_default(),
            )
        })
        .collect::<Vec<_>>();

    Ok(data)
}

#[derive(Clone)]
struct DataSample {
    sample_id: usize,
    image: kornia_rs::image::Image<f32, 3>,
    angular_velocity: f64,
}

struct Dataloader {
    pub images_dir: PathBuf,
    pub batch_size: usize,
    data: Vec<(String, f64)>,
}

impl Dataloader {
    fn new(data_dir: PathBuf, batch_size: usize) -> Self {
        // Load the data from the CSV file
        let data = load_data(&data_dir.join("samples.csv")).unwrap();

        let images_dir = data_dir.join("oak0/left");

        Self {
            images_dir,
            batch_size,
            data,
        }
    }

    // TODO: pass data sample a Genric type
    async fn start<F>(&self, f: F) -> Arc<Mutex<tokio::sync::mpsc::Receiver<Vec<DataSample>>>>
    where
        F: Fn(usize, String, f64) -> DataSample + Send + Sync + 'static,
    {
        let (tx_samples, rx_samples) = tokio::sync::mpsc::channel(self.batch_size);
        let rx_samples = Arc::new(Mutex::new(rx_samples));

        let data = self.data.clone();

        //// this is the data reader thread running in parallel. Ideally this would be a separate
        //// process, but for the sake of simplicity we are using a thread here.
        //// The data reader thread reads the data from the disk and sends it to the batcher thread
        //// We should pass a user function to customize how to read the data or do some pre-processing
        let _sample_sender_thread = tokio::spawn({
            async move {
                let sample_count = AtomicUsize::new(0);
                data.into_par_iter()
                    .for_each(|(img_path, angular_velocity)| {
                        let sample_id =
                            sample_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        tx_samples
                            .blocking_send(f(sample_id, img_path, angular_velocity))
                            .unwrap();
                    });
            }
        });

        // this is the batcher thread which receives the samples from the data reader thread
        // and batches them together before sending them to the next stage
        //let (tx_batcher, mut rx_batcher) = tokio::sync::mpsc::channel(batch_size);
        let (tx_batcher, rx_batcher) = tokio::sync::mpsc::channel(self.batch_size);
        let rx_batcher = Arc::new(Mutex::new(rx_batcher));

        let _batcher_sender_thread = tokio::spawn({
            let batch_size = self.batch_size;
            async move {
                let rx_samples = rx_samples.clone();
                let batch = Arc::new(Mutex::new(Vec::with_capacity(batch_size)));
                while let Some(sample) = rx_samples.lock().await.recv().await {
                    //println!("Received sample id: {:?}", sample.sample_id);
                    let batch = batch.clone();
                    if batch.lock().await.len() < batch_size {
                        batch.lock().await.push(sample);
                    } else {
                        tx_batcher.send(batch.lock().await.clone()).await.unwrap();
                        batch.lock().await.clear();
                        println!("Sent batch");
                    }
                }
            }
        });

        rx_batcher
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    println!("{:?}", args);

    // Create a new dataloader
    let data_loader = Dataloader::new(args.data_dir.clone(), args.batch_size);

    // Start the training loop

    for epoch in 0..args.epochs {
        println!("Epoch: {:?}", epoch);

        // This is the first stage of the pipeline that receives the data samples
        // and prepares them for batching. Here we pass a closure that reads the
        // image from the disk and prepares the sample.
        let batch_rx = data_loader
            .start({
                let images_dir = data_loader.images_dir.clone();
                move |sample_id, img_path, ang_vel| {
                    println!("Preparing sample: {:?}", sample_id);
                    let img_path = images_dir.join(Path::new(&img_path.clone()));
                    let img = F::read_image_jpeg(&img_path).unwrap();
                    let img = img.cast_and_scale::<f32>(1. / 255.0).unwrap();
                    DataSample {
                        sample_id,
                        image: img,
                        angular_velocity: ang_vel,
                    }
                }
            })
            .await;

        // This is the final stage of the pipeline that receives the batches
        // before the data is sent to the model for processing
        let mut batch_count = 0;
        while let Some(batch) = batch_rx.lock().await.recv().await {
            println!(
                "Received batch: {:?} of size: {:?}",
                batch_count,
                batch.len()
            );
            batch_count += 1;
        }
    }

    println!("Done");

    Ok(())
}
