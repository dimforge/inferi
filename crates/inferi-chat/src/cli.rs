use crate::chat_template::ChatTemplate;
use crate::llm::ChatLlm;
use crate::prompt::{ChatEvent, Prompt};
use crate::sampler::SamplerParams;
use async_std::sync::RwLock;
use clap::Parser;
use colored::Colorize;
use inferi::gguf::Gguf;
use inferi::models::segment_anything::SamGgmlFile;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(clap::ValueEnum, Clone, Debug, Default)]
pub enum BackendArg {
    /// Use the WebGPU backend.
    #[cfg(feature = "webgpu")]
    #[default]
    Webgpu,
    /// Use the CUDA backend.
    #[cfg(feature = "cuda")]
    #[cfg_attr(not(feature = "webgpu"), default)]
    Cuda,
    /// Use the CPU backend.
    #[cfg(feature = "cpu")]
    #[cfg_attr(not(any(feature = "webgpu", feature = "cuda")), default)]
    Cpu,
}

#[derive(Parser, Debug)]
#[command(version, about)]
pub struct Cli {
    /// Path to the GGUF model to load.
    pub path: Option<PathBuf>,

    /// Image input for models that need it (like segment-anything).
    #[arg(long)]
    pub image: Option<PathBuf>,

    /// GPU backend to use for inference.
    #[arg(long, value_enum, default_value_t)]
    pub backend: BackendArg,

    /// Initial prompt to send to the model in headless mode.
    #[arg(long)]
    pub prompt: Option<String>,

    /// Exit after the first LLM response instead of continuing the chat.
    #[arg(long, default_value_t = false)]
    pub nochat: bool,

    /// If `true` the app will run without a GUI.
    #[arg(long, default_value_t = false)]
    pub headless: bool,

    /// If `true`, details of the GGUF file will be printed.
    #[arg(long, default_value_t = false)]
    pub inspect: bool,
}

pub async fn run_headless(cli: &Cli) -> anyhow::Result<()> {
    let gpu = crate::init_gpu_with_backend(&cli.backend).await?;
    let Some(path) = &cli.path else {
        println!("{}", "No model file provided, exiting.".red());
        return Ok(());
    };

    println!("{}", format!("Loading GGUF file: {:?}", cli.path).dimmed());
    let t_gguf = web_time::Instant::now();
    let file = File::open(path).expect("Unable to open the GGUF model file");
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let gguf = {
        if path.extension().map(|ext| ext.to_str().unwrap()) == Some("bin") {
            println!("Attempting to load file as legacy GGML format.");
            SamGgmlFile::from_bytes(&mmap[..])
                .map(|ggml| Ok(ggml.into_gguf()))
                .unwrap_or_else(|e| {
                    println!("Error loading legacy GGML format: {}", e);
                    println!("Trying GGUF loader.");
                    Gguf::from_bytes(&mmap[..])
                })?
        } else {
            Gguf::from_bytes(&mmap[..])?
        }
    };
    println!(
        "{}",
        format!(
            "GGUF model loaded in {:.2} seconds.",
            t_gguf.elapsed().as_secs_f32()
        )
        .dimmed()
    );

    if cli.inspect {
        gguf.print_metadata();
        gguf.print_tensors();
    }

    let t_chat_llm = web_time::Instant::now();
    let llm = Arc::new(RwLock::new(ChatLlm::from_gguf(&gpu.backend, &gguf).await?));
    let chat_template = ChatTemplate::from_gguf(&gguf);
    println!(
        "{}",
        format!(
            "Uploaded model to GPU in {:.2} seconds.",
            t_chat_llm.elapsed().as_secs_f32()
        )
        .dimmed()
    );

    println!("{}", "Starting interactive chat:".dimmed());
    let mut prompt = Prompt::default();
    let mut next_pos = 0;
    let mut tok_per_second = 0.0;
    let sampler = SamplerParams::default();
    let mut initial_prompt = cli.prompt.clone();

    loop {
        let user_prompt = if let Some(initial) = initial_prompt.take() {
            println!("{}", "[User]".purple().bold());
            println!("{initial}");
            initial
        } else {
            // Read stdin.
            println!("{}", "[User]".purple().bold());
            let mut input = String::new();
            std::io::stdout().flush()?;
            std::io::stdin().read_line(&mut input)?;
            input.truncate(input.trim_end().len());

            // Check for exit command
            if input.trim() == "exit" {
                println!("{}", "Exiting...".dimmed());
                return Ok(());
            }
            input
        };

        prompt.append_user(user_prompt);

        // Forward the transformer.
        let (snd, rcv) = async_channel::unbounded();

        {
            let gpu = gpu.clone();
            let llm = llm.clone();
            let prompt = prompt.clone();
            let chat_template = chat_template.clone();
            async_std::task::spawn(async move {
                llm.write()
                    .await
                    .forward(
                        &gpu.backend,
                        prompt,
                        sampler,
                        chat_template,
                        next_pos,
                        |msg| Ok(snd.send_blocking(msg)?),
                    )
                    .await
            });
        }

        let mut full_response = String::new();
        let mut last_tok = String::new();
        println!("{}", "[Assistant]".green().bold());

        while let Ok(event) = rcv.recv().await {
            if let ChatEvent::Token {
                string,
                next_pos: next,
                token_count,
                token_time,
            } = event
            {
                let tps = token_count as f64 / token_time;

                // Don't print multiple newlines, takes too
                // much room on the console.
                if last_tok != "\n" || string != "\n" {
                    print!("{}", string);
                }
                next_pos = next;
                tok_per_second = tps;
                full_response.push_str(&string);
                last_tok = string;
                std::io::stdout().flush()?;
            }
        }
        println!();
        println!(
            "{}",
            format!(
                "({:.2} tok/s) − generated {} tokens",
                tok_per_second, next_pos
            )
            .italic()
            .dimmed()
        );

        if cli.nochat {
            return Ok(());
        }

        prompt.append_assistant(full_response);
    }
}
