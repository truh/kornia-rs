use std::path::Path;

use crate::image::{Image, ImageSize};
use anyhow::Result;
use gst::{buffer, prelude::*};

/// Extracts an image frame from the appsink
///
/// # Arguments
///
/// * `appsink` - The AppSink
///
/// # Returns
///
/// An image frame
fn extract_image_frame(appsink: &gst_app::AppSink) -> Result<Option<Image<u8, 3>>> {
    let sample = match appsink.pull_sample() {
        Ok(sample) => sample,
        Err(_) => return Ok(None),
    };
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
    let img = Image::<u8, 3>::new(ImageSize { width, height }, map.as_slice().to_vec())?;
    Ok(Some(img))
}

pub struct VideoReader {
    pipeline: gst::Pipeline,
    appsink: gst_app::AppSink,
}

impl VideoReader {
    pub fn new(file_path: &Path) -> Result<Self> {
        gst::init()?;

        let pipeline_str = format!(
            "filesrc location={} ! decodebin ! videoconvert ! video/x-raw,format=RGB ! appsink name=sink",
            file_path.to_str().unwrap()
        );

        let pipeline = gst::parse::launch(&pipeline_str)?
            .downcast::<gst::Pipeline>()
            .map_err(|_| anyhow::anyhow!("Failed to downcast pipeline"))?;

        let appsink = pipeline
            .by_name("sink")
            .ok_or_else(|| anyhow::anyhow!("Failed to get sink"))?
            .dynamic_cast::<gst_app::AppSink>()
            .map_err(|_| anyhow::anyhow!("Failed to cast to AppSink"))?;

        Ok(Self { pipeline, appsink })
    }

    pub fn start(&self) -> Result<()> {
        self.pipeline.set_state(gst::State::Playing)?;
        Ok(())
    }

    pub fn stop(&self) -> Result<()> {
        self.pipeline.set_state(gst::State::Null)?;
        Ok(())
    }

    pub fn grab_frame(&self) -> Result<Option<Image<u8, 3>>> {
        let appsink = &self
            .appsink
            .clone()
            .dynamic_cast::<gst_app::AppSink>()
            .map_err(|_| anyhow::anyhow!("Failed to cast to AppSink"))?;

        extract_image_frame(appsink)
    }
}

pub struct VideoWriter {
    pipeline: gst::Pipeline,
    appsrc: gst_app::AppSrc,
    counter: u32,
    fps: f32,
    gst_caps: gst::Caps,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl VideoWriter {
    pub fn new(file_path: &Path, fps: f32, width: usize, height: usize) -> Result<Self> {
        gst::init()?;

        let pipeline_str = format!(
            "appsrc name=src caps=video/x-raw,format=RGB,width={width},height={height},framerate={fps}/1 !
            x264enc ! mp4mux ! filesink location={} ",
            file_path.to_str().unwrap(),
        );

        let pipeline = gst::parse::launch(&pipeline_str)?
            .downcast::<gst::Pipeline>()
            .map_err(|_| anyhow::anyhow!("Failed to downcast pipeline"))?;

        let appsrc = pipeline
            .by_name("src")
            .ok_or_else(|| anyhow::anyhow!("Failed to get src"))?
            .dynamic_cast::<gst_app::AppSrc>()
            .map_err(|_| anyhow::anyhow!("Failed to cast to AppSrc"))?;

        appsrc.set_property("format", gst::Format::Time);
        //appsrc.set_property("block", &true);
        appsrc.set_property("is-live", &true);

        let gst_caps = gst::Caps::new_empty_simple("timestamp");

        Ok(Self {
            pipeline,
            appsrc,
            counter: 0,
            fps: fps,
            gst_caps: gst_caps,
            handle: None,
        })
    }

    pub fn start(&mut self) -> Result<()> {
        self.pipeline.set_state(gst::State::Playing)?;
        let bus = self.pipeline.bus().unwrap();
        let handle = std::thread::spawn(move || {
            for msg in bus.iter_timed(gst::ClockTime::NONE) {
                match msg.view() {
                    gst::MessageView::Eos(..) => {
                        break;
                    }
                    gst::MessageView::Error(err) => {
                        break;
                    }
                    _ => (),
                }
            }
        });
        self.handle = Some(handle);
        Ok(())
    }

    pub fn stop(&mut self) -> Result<()> {
        self.appsrc.end_of_stream()?;
        self.pipeline.set_state(gst::State::Null)?;
        self.handle.take().unwrap().join().unwrap();
        Ok(())
    }

    pub fn write(&mut self, img: &Image<u8, 3>) -> Result<()> {
        //let mut buffer = gst::Buffer::with_size(img.data.len())?;
        //buffer
        //    .get_mut()
        //    .unwrap()
        //    .map_writable()?
        //    .as_mut_slice()
        //    .copy_from_slice(&img.data.as_slice().unwrap());
        let mut buffer =
            gst::Buffer::from_mut_slice(img.data.as_slice().expect("Failed to get data").to_vec());

        let time =
            gst::ClockTime::from_nseconds(self.counter as u64 * 1_000_000_000 / self.fps as u64);

        //buffer
        //    .get_mut()
        //    .unwrap()
        //    .set_pts(gst::ClockTime::from_nseconds(
        //        self.counter as u64 * 1_000_000_000 / self.fps as u64,
        //    ));
        let _ = gst::meta::ReferenceTimestampMeta::add(
            buffer.get_mut().unwrap(),
            &self.gst_caps.clone(),
            time,
            time,
        );

        self.counter += 1;

        self.appsrc.push_buffer(buffer)?;

        Ok(())
    }
}
