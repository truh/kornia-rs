use crate::image::{Image, ImageSize};
use anyhow::Result;
use gst::prelude::*;

pub struct StreamCapture {
    pipeline: gst::Pipeline,
    receiver: tokio::sync::mpsc::Receiver<Image<u8, 3>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl StreamCapture {
    pub fn new(pipeline_str: &str) -> Result<Self> {
        gst::init()?;

        let pipeline = gst::parse::launch(&pipeline_str)?
            .downcast::<gst::Pipeline>()
            .expect("not a pipeline");

        let appsink = pipeline
            .by_name("sink")
            .ok_or_else(|| anyhow::anyhow!("Failed to get sink"))?
            .dynamic_cast::<gst_app::AppSink>()
            .map_err(|_| anyhow::anyhow!("Failed to cast to AppSink"))?;

        let (tx, rx) = tokio::sync::mpsc::channel(50);

        appsink.set_callbacks(
            gst_app::AppSinkCallbacks::builder()
                .new_sample(move |sink| match Self::extract_image_frame(sink) {
                    Ok(frame) => {
                        println!("Received frame");
                        if tx.blocking_send(frame).is_err() {
                            Err(gst::FlowError::Error)
                        } else {
                            Ok(gst::FlowSuccess::Ok)
                        }
                    }
                    Err(_) => Err(gst::FlowError::Error),
                })
                .build(),
        );

        Ok(Self {
            pipeline,
            receiver: rx,
            handle: None,
        })
    }

    /// Extracts an image frame from the appsink
    ///
    /// # Arguments
    ///
    /// * `appsink` - The AppSink
    ///
    /// # Returns
    ///
    /// An image frame
    fn extract_image_frame(appsink: &gst_app::AppSink) -> Result<Image<u8, 3>> {
        let sample = appsink.pull_sample()?;
        let caps = sample
            .caps()
            .ok_or_else(|| anyhow::anyhow!("Failed to get caps from sample"))?;
        let structure = caps
            .structure(0)
            .ok_or_else(|| anyhow::anyhow!("Failed to get structure"))?;
        let height = structure.get::<i32>("height")? as usize;
        let width = structure.get::<i32>("width")? as usize;

        let buffer = sample
            .buffer()
            .ok_or_else(|| anyhow::anyhow!("Failed to get buffer from sample"))?;
        let map = buffer.map_readable()?;
        Image::<u8, 3>::new(ImageSize { width, height }, map.as_slice().to_vec())
    }

    /// Runs the webcam capture object and grabs frames from the camera
    ///
    /// # Arguments
    ///
    /// * `f` - A function that takes an image frame
    pub async fn run<F>(&mut self, f: F) -> Result<()>
    where
        F: Fn(Image<u8, 3>) -> Result<()>,
    {
        // start the pipeline
        let pipeline = &self.pipeline;
        pipeline.set_state(gst::State::Playing)?;

        let bus = pipeline
            .bus()
            .ok_or_else(|| anyhow::anyhow!("Failed to get bus"))?;

        // start a thread to handle the messages from the bus
        let handle = std::thread::spawn(move || {
            for msg in bus.iter_timed(gst::ClockTime::NONE) {
                use gst::MessageView;
                match msg.view() {
                    MessageView::Eos(..) => break,
                    MessageView::Error(err) => {
                        eprintln!(
                            "Error from {:?}: {} ({:?})",
                            msg.src().map(|s| s.path_string()),
                            err.error(),
                            err.debug()
                        );
                        break;
                    }
                    _ => (),
                }
            }
        });
        self.handle = Some(handle);

        // start grabbing frames from the camera
        while let Some(img) = self.receiver.recv().await {
            f(img)?;
        }

        Ok(())
    }
}
