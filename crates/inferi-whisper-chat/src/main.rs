use anyhow::Result;
use clap::Parser;
use colored::Colorize;
use inferi::context::{LlmContext, LlmOps};
use inferi::gguf::Gguf;
use inferi::models::whisper;
use inferi::models::whisper::model::{ModelLoader, Whisper};
use inferi::models::whisper::WhisperConfig;
use inferi::re_exports::safetensors::SafeTensors;
use inferi::re_exports::tokenizers::Tokenizer;
use inferi::re_exports::vortx::shapes::TensorLayoutBuffers;
use inferi::re_exports::vortx::tensor::TensorBuilder;
use inferi::tensor_cache::TensorCache;
use khal::backend::{GpuBackend, WebGpu};
use khal::BufferUsages;
use std::fs::File;
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::Arc;

mod chat_llama2;
mod chat_template;
mod chat_whisper;
mod llm;
mod mic;
mod prompt;
mod sampler;

use chat_template::ChatTemplate;
use chat_whisper::{Model, WhisperChat};
use llm::ChatLlm;
use prompt::{ChatEvent, Prompt};
use sampler::SamplerParams;

#[derive(Parser, Debug)]
#[command(version, about = "Voice-to-LLM chat interface")]
pub struct Cli {
    /// Path to the Whisper model directory (containing config.json and model.safetensors).
    #[arg(long)]
    pub whisper: PathBuf,

    /// Path to the LLM model (GGUF format).
    #[arg(long)]
    pub llm: PathBuf,
}

#[derive(Clone)]
pub struct GpuInstanceCtx {
    pub backend: Arc<GpuBackend>,
}

impl GpuInstanceCtx {
    pub fn new(backend: GpuBackend) -> Self {
        Self {
            backend: Arc::new(backend),
        }
    }
}

async fn init_wgpu() -> Result<GpuInstanceCtx> {
    let features = wgpu::Features::default();
    let limits = wgpu::Limits {
        max_buffer_size: 2_000_000_000,
        max_storage_buffer_binding_size: 2_000_000_000,
        ..Default::default()
    };

    let mut webgpu = WebGpu::new(features, limits).await?;
    webgpu.force_buffer_copy_src = true;
    let backend = GpuBackend::WebGpu(webgpu);
    Ok(GpuInstanceCtx::new(backend))
}

fn wait_for_enter(message: &str) {
    print!("{}", message);
    std::io::stdout().flush().unwrap();
    let mut input = String::new();
    std::io::stdin().lock().read_line(&mut input).unwrap();
}

/// Records audio until Enter is pressed, returning the path to the recording.
fn record_until_enter(recording_path: &str) -> Result<()> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("No default input device available"))?;

    println!("{}", format!("Recording from: {}", device.name()?).dimmed());

    let supported_config = device.default_input_config()?;
    let sample_format = supported_config.sample_format();
    let native_sample_rate = supported_config.sample_rate().0;
    let channels = supported_config.channels();
    let config: cpal::StreamConfig = supported_config.into();

    const TARGET_SAMPLE_RATE: u32 = 16000;

    // Output WAV spec at 16kHz mono
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: TARGET_SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let writer = hound::WavWriter::create(recording_path, spec)?;
    let writer = Arc::new(Mutex::new(Some(writer)));
    let writer_clone = writer.clone();

    let stop_flag = Arc::new(AtomicBool::new(false));
    let stop_flag_clone = stop_flag.clone();

    let err_fn = |err| eprintln!("Recording error: {}", err);

    // Resampling state
    let resample_ratio = native_sample_rate as f64 / TARGET_SAMPLE_RATE as f64;
    let sample_accumulator = Arc::new(Mutex::new(Vec::<f32>::new()));
    let sample_accumulator_clone = sample_accumulator.clone();
    let input_sample_index = Arc::new(Mutex::new(0.0f64));
    let input_sample_index_clone = input_sample_index.clone();

    let process_samples = move |samples: &[f32]| {
        if stop_flag_clone.load(Ordering::Relaxed) {
            return;
        }

        // Convert to mono by averaging channels
        let mono_samples: Vec<f32> = samples
            .chunks(channels as usize)
            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
            .collect();

        let mut acc = sample_accumulator_clone.lock().unwrap();
        acc.extend(mono_samples);

        let mut idx = input_sample_index_clone.lock().unwrap();

        if let Ok(mut guard) = writer_clone.lock() {
            if let Some(ref mut writer) = *guard {
                // Simple linear interpolation resampling
                while *idx < acc.len() as f64 {
                    let i = *idx as usize;
                    let frac = *idx - i as f64;

                    let sample = if i + 1 < acc.len() {
                        acc[i] * (1.0 - frac as f32) + acc[i + 1] * frac as f32
                    } else {
                        acc[i]
                    };

                    let sample_i16 = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                    let _ = writer.write_sample(sample_i16);
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
    };

    let stream = match sample_format {
        cpal::SampleFormat::I16 => device.build_input_stream(
            &config,
            {
                let process = process_samples.clone();
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    let samples: Vec<f32> =
                        data.iter().map(|&s| s as f32 / i16::MAX as f32).collect();
                    process(&samples);
                }
            },
            err_fn,
            None,
        )?,
        cpal::SampleFormat::F32 => device.build_input_stream(
            &config,
            {
                let process = process_samples.clone();
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    process(data);
                }
            },
            err_fn,
            None,
        )?,
        cpal::SampleFormat::U16 => device.build_input_stream(
            &config,
            {
                let process = process_samples.clone();
                move |data: &[u16], _: &cpal::InputCallbackInfo| {
                    let samples: Vec<f32> = data
                        .iter()
                        .map(|&s| (s as f32 - 32768.0) / 32768.0)
                        .collect();
                    process(&samples);
                }
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

    // Wait for Enter to stop recording
    wait_for_enter(&format!(
        "{}\n",
        "Recording... Press Enter to stop.".yellow()
    ));

    stop_flag.store(true, Ordering::Relaxed);
    drop(stream);

    // Finalize the WAV file
    let mut guard = writer.lock().unwrap();
    if let Some(writer) = guard.take() {
        writer.finalize()?;
    }

    println!("{}", "Recording saved.".green());
    Ok(())
}

async fn run(cli: &Cli) -> Result<()> {
    println!("{}", "Initializing GPU...".dimmed());
    let gpu = init_wgpu().await?;

    // Load Whisper model from directory
    println!(
        "{}",
        format!("Loading Whisper model from {:?}...", cli.whisper).dimmed()
    );
    let whisper_config_path = cli.whisper.join("config.json");
    let whisper_model_path = cli.whisper.join("model.safetensors");
    let whisper_data = std::fs::read(&whisper_model_path)
        .map_err(|e| anyhow::anyhow!("Failed to read {:?}: {}", whisper_model_path, e))?;
    let whisper_config: WhisperConfig = serde_json::from_str(
        &std::fs::read_to_string(&whisper_config_path)
            .map_err(|e| anyhow::anyhow!("Failed to read {:?}: {}", whisper_config_path, e))?,
    )?;
    let whisper_st = SafeTensors::deserialize(&whisper_data)?;
    let whisper_loader = ModelLoader::new(&gpu.backend, &whisper_st);
    let whisper_model = Whisper::new(&whisper_loader, whisper_config.clone())?;
    println!("{}", "Whisper model loaded.".green());

    // Load mel filters (embedded at compile time)
    let mel_bytes = include_bytes!("../../inferi/src/models/whisper/melfilters.bytes");
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(mel_bytes, &mut mel_filters);

    // Load whisper tokenizer
    let whisper_tokenizer_path = cli.whisper.join("tokenizer.json");
    let whisper_tokenizer = Tokenizer::from_file(&whisper_tokenizer_path).map_err(|e| {
        anyhow::anyhow!(
            "Failed to load whisper tokenizer from {:?}: {}",
            whisper_tokenizer_path,
            e
        )
    })?;

    // Create WhisperChat once (reused for all transcriptions)
    // Using English language token by default
    let lang_token = chat_whisper::token_id(&whisper_tokenizer, "<|en|>").ok();
    let mut whisper_chat = WhisperChat::new(
        Model::Normal(whisper_model),
        whisper_tokenizer,
        299792458,
        &gpu.backend,
        lang_token,
        None,
        false,
        None,
        false,
    )?;

    // Create reusable LlmOps
    let whisper_ops = LlmOps::new(&gpu.backend)?;

    // Load LLM model
    println!(
        "{}",
        format!("Loading LLM model from {:?}...", cli.llm).dimmed()
    );
    let t_gguf = web_time::Instant::now();
    let llm_file = File::open(&cli.llm)?;
    let llm_mmap = unsafe { memmap2::Mmap::map(&llm_file)? };
    let gguf = Gguf::from_bytes(&llm_mmap[..])?;
    println!(
        "{}",
        format!(
            "GGUF model loaded in {:.2} seconds.",
            t_gguf.elapsed().as_secs_f32()
        )
        .dimmed()
    );

    let t_upload = web_time::Instant::now();
    let llm = Arc::new(ChatLlm::from_gguf(&gpu.backend, &gguf).await?);
    let chat_template = ChatTemplate::from_gguf(&gguf);
    println!(
        "{}",
        format!(
            "Uploaded LLM to GPU in {:.2} seconds.",
            t_upload.elapsed().as_secs_f32()
        )
        .dimmed()
    );

    println!("\n{}", "=".repeat(50).dimmed());
    println!("{}", "Voice-to-LLM Chat Ready!".green().bold());
    println!("{}", "=".repeat(50).dimmed());

    let recording_path = "/tmp/inferi_whisper_recording.wav";
    let mut prompt = Prompt::default();
    let sampler = SamplerParams::default();
    let gpu = Arc::new(gpu);

    // Reusable buffers for whisper context
    let mut whisper_cache = TensorCache::default();
    let mut whisper_shapes = TensorLayoutBuffers::new(&gpu.backend);
    let mut llm_next_pos = 0;

    loop {
        // Wait for user to start recording
        println!();
        wait_for_enter(&format!(
            "{}\n",
            "Press Enter to start recording...".cyan().bold()
        ));

        // Record until Enter is pressed
        record_until_enter(recording_path)?;

        // Process recording with Whisper
        println!("{}", "Transcribing...".yellow());
        let (pcm_data, sample_rate) = candle_examples::audio::pcm_decode(recording_path)?;
        if sample_rate != whisper::SAMPLE_RATE as u32 {
            anyhow::bail!(
                "Recording must have {} Hz sample rate, got {}",
                whisper::SAMPLE_RATE,
                sample_rate
            );
        }

        let mel = whisper::audio::pcm_to_mel(&whisper_config, &pcm_data, &mel_filters);
        let mel_len = mel.len();
        let mel_shape = [
            1,
            whisper_config.num_mel_bins as u32,
            (mel_len / whisper_config.num_mel_bins) as u32,
        ];
        let mel_tensor = TensorBuilder::tensor(&mel_shape, BufferUsages::STORAGE)
            .build_init(&gpu.backend, &mel)?;

        // Clear caches for fresh transcription
        whisper_cache.clear();
        whisper_shapes.clear_tmp();

        let mut ctx = LlmContext {
            backend: &gpu.backend,
            cache: &mut whisper_cache,
            shapes: &mut whisper_shapes,
            pass: None,
            encoder: None,
            ops: &whisper_ops,
        };

        let segments = whisper_chat.run(&mut ctx, &mel_tensor).await?;

        // Collect transcribed text
        let transcribed_text: String = segments
            .iter()
            .map(|s| s.dr.text.trim().to_string())
            .collect::<Vec<_>>()
            .join(" ");

        if transcribed_text.is_empty() {
            println!("{}", "No speech detected. Try again.".red());
            continue;
        }

        println!(
            "{}",
            format!("[You said]: {}", transcribed_text).purple().bold()
        );

        // Add to conversation and get LLM response
        prompt.append_user(transcribed_text);

        let (snd, rcv) = async_channel::unbounded();
        let llm_clone = llm.clone();
        let gpu_clone = gpu.clone();
        let prompt_clone = prompt.clone();
        let chat_template_clone = chat_template.clone();

        // Always start from position 0 - the full conversation is re-tokenized each turn
        async_std::task::spawn(async move {
            llm_clone
                .forward(
                    &gpu_clone.backend,
                    prompt_clone,
                    sampler,
                    chat_template_clone,
                    llm_next_pos,
                    |msg| Ok(snd.send_blocking(msg)?),
                )
                .await
        });

        let mut full_response = String::new();
        let mut last_tok = String::new();
        let mut tok_per_second = 0.0;
        let mut token_count = 0;
        println!("{}", "[Assistant]:".green().bold());

        while let Ok(event) = rcv.recv().await {
            match event {
                ChatEvent::Token {
                    string,
                    token_count: count,
                    token_time,
                    ..
                } => {
                    let tps = count as f64 / token_time;

                    if last_tok != "\n" || string != "\n" {
                        print!("{}", string);
                    }
                    token_count = count;
                    tok_per_second = tps;
                    full_response.push_str(&string);
                    last_tok = string;
                    std::io::stdout().flush()?;
                }
                ChatEvent::Finish { next_pos } => {
                    llm_next_pos = next_pos;
                }
                _ => {}
            }
        }

        println!();
        println!(
            "{}",
            format!(
                "({:.2} tok/s) - generated {} tokens",
                tok_per_second, token_count
            )
            .italic()
            .dimmed()
        );

        prompt.append_assistant(full_response);
    }
}

fn main() {
    let cli = Cli::parse();
    futures::executor::block_on(run(&cli)).unwrap();
}
