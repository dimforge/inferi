#![allow(dead_code)]

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex};

/// Target sample rate for Whisper compatibility.
const TARGET_SAMPLE_RATE: u32 = 16000;

/// Lists all available input (microphone) devices.
/// Returns a vector of (index, device_name) pairs.
pub fn list_input_devices() -> anyhow::Result<Vec<(usize, String)>> {
    let host = cpal::default_host();
    let devices: Vec<_> = host
        .input_devices()?
        .enumerate()
        .filter_map(|(i, d)| d.name().ok().map(|name| (i, name)))
        .collect();
    Ok(devices)
}

/// Gets the default input device.
pub fn default_input_device() -> anyhow::Result<cpal::Device> {
    let host = cpal::default_host();
    host.default_input_device()
        .ok_or_else(|| anyhow::anyhow!("No default input device available"))
}

/// Gets an input device by index.
pub fn get_input_device_by_index(index: usize) -> anyhow::Result<cpal::Device> {
    let host = cpal::default_host();
    host.input_devices()?
        .nth(index)
        .ok_or_else(|| anyhow::anyhow!("Input device at index {} not found", index))
}

/// Gets an input device by name (partial match).
pub fn get_input_device_by_name(name: &str) -> anyhow::Result<cpal::Device> {
    let host = cpal::default_host();
    for device in host.input_devices()? {
        if let Ok(device_name) = device.name() {
            if device_name.to_lowercase().contains(&name.to_lowercase()) {
                return Ok(device);
            }
        }
    }
    Err(anyhow::anyhow!(
        "No input device found matching name: {}",
        name
    ))
}

/// Records audio from the given device to a WAV file at 16kHz sample rate.
///
/// The audio is recorded at the device's native sample rate and resampled to 16kHz
/// for Whisper compatibility.
///
/// # Arguments
/// * `device` - The input device to record from
/// * `output_path` - Path to the output WAV file
/// * `duration_secs` - Duration of recording in seconds
pub fn record_to_file(
    device: &cpal::Device,
    output_path: &str,
    duration_secs: u32,
) -> anyhow::Result<()> {
    // Try to get a config that supports 16kHz, otherwise use default and resample
    let supported_config = device.default_input_config()?;
    let sample_format = supported_config.sample_format();
    let native_sample_rate = supported_config.sample_rate().0;
    let channels = supported_config.channels();

    // Build config with native sample rate (we'll resample later)
    let config: cpal::StreamConfig = supported_config.into();

    // Output WAV spec at 16kHz mono
    let spec = hound::WavSpec {
        channels: 1, // Mono for Whisper
        sample_rate: TARGET_SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let writer = hound::WavWriter::create(output_path, spec)?;
    let writer = Arc::new(Mutex::new(Some(writer)));
    let writer_clone = writer.clone();

    let err_fn = |err| eprintln!("Recording error: {}", err);

    // Calculate total output samples at 16kHz mono
    let total_output_samples = (TARGET_SAMPLE_RATE * duration_secs) as usize;
    let samples_written = Arc::new(Mutex::new(0usize));
    let samples_written_clone = samples_written.clone();

    // Resampling state
    let resample_ratio = native_sample_rate as f64 / TARGET_SAMPLE_RATE as f64;
    let sample_accumulator = Arc::new(Mutex::new(Vec::<f32>::new()));
    let sample_accumulator_clone = sample_accumulator.clone();
    let input_sample_index = Arc::new(Mutex::new(0.0f64));
    let input_sample_index_clone = input_sample_index.clone();

    let stream = match sample_format {
        cpal::SampleFormat::I16 => device.build_input_stream(
            &config,
            move |data: &[i16], _: &cpal::InputCallbackInfo| {
                process_samples(
                    data.iter().map(|&s| s as f32 / i16::MAX as f32),
                    channels,
                    resample_ratio,
                    total_output_samples,
                    &writer_clone,
                    &samples_written_clone,
                    &sample_accumulator_clone,
                    &input_sample_index_clone,
                );
            },
            err_fn,
            None,
        )?,
        cpal::SampleFormat::F32 => device.build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                process_samples(
                    data.iter().copied(),
                    channels,
                    resample_ratio,
                    total_output_samples,
                    &writer_clone,
                    &samples_written_clone,
                    &sample_accumulator_clone,
                    &input_sample_index_clone,
                );
            },
            err_fn,
            None,
        )?,
        cpal::SampleFormat::U16 => device.build_input_stream(
            &config,
            move |data: &[u16], _: &cpal::InputCallbackInfo| {
                process_samples(
                    data.iter().map(|&s| (s as f32 - 32768.0) / 32768.0),
                    channels,
                    resample_ratio,
                    total_output_samples,
                    &writer_clone,
                    &samples_written_clone,
                    &sample_accumulator_clone,
                    &input_sample_index_clone,
                );
            },
            err_fn,
            None,
        )?,
        _ => {
            return Err(anyhow::anyhow!(
                "Unsupported sample format: {:?}",
                sample_format
            ))
        }
    };

    stream.play()?;
    std::thread::sleep(std::time::Duration::from_secs(duration_secs as u64));
    drop(stream);

    // Finalize the WAV file
    let mut guard = writer.lock().unwrap();
    if let Some(writer) = guard.take() {
        writer.finalize()?;
    }

    Ok(())
}

/// Process incoming samples: convert to mono, resample to 16kHz, and write to WAV.
#[allow(clippy::too_many_arguments)]
fn process_samples<I: Iterator<Item = f32>>(
    samples: I,
    channels: u16,
    resample_ratio: f64,
    total_output_samples: usize,
    writer: &Arc<Mutex<Option<hound::WavWriter<std::io::BufWriter<std::fs::File>>>>>,
    samples_written: &Arc<Mutex<usize>>,
    sample_accumulator: &Arc<Mutex<Vec<f32>>>,
    input_sample_index: &Arc<Mutex<f64>>,
) {
    let samples: Vec<f32> = samples.collect();

    // Convert to mono by averaging channels
    let mono_samples: Vec<f32> = samples
        .chunks(channels as usize)
        .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
        .collect();

    let mut acc = sample_accumulator.lock().unwrap();
    acc.extend(mono_samples);

    let mut count = samples_written.lock().unwrap();
    let mut idx = input_sample_index.lock().unwrap();

    if let Ok(mut guard) = writer.lock() {
        if let Some(ref mut writer) = *guard {
            // Simple linear interpolation resampling
            while *idx < acc.len() as f64 && *count < total_output_samples {
                let i = *idx as usize;
                let frac = *idx - i as f64;

                let sample = if i + 1 < acc.len() {
                    acc[i] * (1.0 - frac as f32) + acc[i + 1] * frac as f32
                } else {
                    acc[i]
                };

                let sample_i16 = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                let _ = writer.write_sample(sample_i16);
                *count += 1;
                *idx += resample_ratio;
            }

            // Remove processed samples from accumulator
            let consumed = *idx as usize;
            if consumed > 0 && consumed <= acc.len() {
                acc.drain(0..consumed);
                *idx -= consumed as f64;
            }
        }
    }
}

/// Records audio from the default input device to a WAV file.
pub fn record_to_file_default(output_path: &str, duration_secs: u32) -> anyhow::Result<()> {
    let device = default_input_device()?;
    println!("Recording from default device {}...", device.name()?);
    record_to_file(&device, output_path, duration_secs)
}

/// Records audio from a device selected by index to a WAV file.
pub fn record_to_file_by_index(
    device_index: usize,
    output_path: &str,
    duration_secs: u32,
) -> anyhow::Result<()> {
    let device = get_input_device_by_index(device_index)?;
    record_to_file(&device, output_path, duration_secs)
}

/// Records audio from a device selected by name to a WAV file.
pub fn record_to_file_by_name(
    device_name: &str,
    output_path: &str,
    duration_secs: u32,
) -> anyhow::Result<()> {
    let device = get_input_device_by_name(device_name)?;
    record_to_file(&device, output_path, duration_secs)
}
