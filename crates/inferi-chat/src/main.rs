use crate::chat_template::ChatTemplate;
use crate::llm::ChatLlm;
use crate::prompt::Prompt;
use crate::sampler::SamplerParams;
use components::{Chat, Home};
use dioxus::prelude::*;
#[cfg(feature = "cuda")]
use khal::backend::Cuda;
use khal::backend::GpuBackend;
#[cfg(feature = "webgpu")]
use khal::backend::WebGpu;
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use crate::cli::Cli;

// mod chat_gpt2;
mod chat_llama2;
mod chat_template;
mod components;
mod llm;
mod prompt;
mod sampler;

#[cfg(not(target_arch = "wasm32"))]
mod cli;
#[cfg(not(target_arch = "wasm32"))]
mod mic;
mod segment_anything;

#[derive(Copy, Clone)]
pub struct UnsupportedBackend;

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

pub type LoadedModelSignal = Signal<Option<LoadedModel>>;

#[derive(Clone, Debug, Default)]
enum PromptResponse {
    #[default]
    Empty,
    Thinking,
    Responding(String),
}

#[derive(Default)]
struct PromptState {
    prompt: Prompt,
    response: PromptResponse,
}

#[derive(Clone)]
pub struct GgufMetadata {
    metadata: Vec<String>,
    tensors: Vec<String>,
}

#[derive(Clone)]
pub struct LoadedModel {
    pub llm: Arc<ChatLlm>,
    pub sampler: SamplerParams,
    pub template: ChatTemplate,
    pub metadata: GgufMetadata,
}

const FAVICON: Asset = asset!("/assets/inferi-logo.png");
const MAIN_CSS: Asset = asset!("/assets/styling/main.css");

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use clap::Parser;
        let cli = Cli::parse();
        let _ = SELECTED_BACKEND.set(cli.backend.clone());
        if cli.headless {
            futures::executor::block_on(cli::run_headless(&cli)).unwrap();
            return;
        }
    }

    #[cfg(feature = "desktop")]
    {
        use dioxus::desktop::{tao, LogicalSize};
        let window =
            tao::window::WindowBuilder::default().with_inner_size(LogicalSize::new(1300.0, 900.0));
        dioxus::LaunchBuilder::new()
            .with_cfg(dioxus::desktop::Config::new().with_window(window))
            .launch(App);
    }

    #[cfg(not(feature = "desktop"))]
    {
        dioxus::launch(App);
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) async fn init_gpu_with_backend(
    #[allow(unused)] backend_arg: &cli::BackendArg,
) -> anyhow::Result<GpuInstanceCtx> {
    let backend = match backend_arg {
        #[cfg(feature = "webgpu")]
        cli::BackendArg::Webgpu => {
            let features = wgpu::Features::default();
            let limits = wgpu::Limits {
                max_buffer_size: 2_000_000_000,
                max_storage_buffer_binding_size: 2_000_000_000,
                ..Default::default()
            };
            let mut webgpu = WebGpu::new(features, limits).await?;
            webgpu.force_buffer_copy_src = true;
            GpuBackend::WebGpu(webgpu)
        }
        #[cfg(feature = "cuda")]
        cli::BackendArg::Cuda => {
            let cuda = Cuda::new(0)?;
            GpuBackend::Cuda(cuda)
        }
        #[cfg(feature = "cpu")]
        cli::BackendArg::Cpu => GpuBackend::Cpu,
    };
    Ok(GpuInstanceCtx::new(backend))
}

#[cfg(not(target_arch = "wasm32"))]
static SELECTED_BACKEND: std::sync::OnceLock<cli::BackendArg> = std::sync::OnceLock::new();

async fn init_gpu() -> anyhow::Result<GpuInstanceCtx> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let backend_arg = SELECTED_BACKEND.get().cloned().unwrap_or_default();
        return init_gpu_with_backend(&backend_arg).await;
    }

    #[cfg(target_arch = "wasm32")]
    {
        let features = wgpu::Features::default();
        let limits = wgpu::Limits {
            max_buffer_size: 2_000_000_000,
            max_storage_buffer_binding_size: 2_000_000_000,
            ..Default::default()
        };
        let mut webgpu = WebGpu::new(features, limits).await?;
        webgpu.force_buffer_copy_src = true;
        Ok(GpuInstanceCtx::new(GpuBackend::WebGpu(webgpu)))
    }
}

#[component]
fn App() -> Element {
    let gpu = use_resource(init_gpu);

    match &*gpu.read_unchecked() {
        Some(Ok(gpu)) => {
            use_context_provider(|| gpu.clone());
            use_context_provider(|| LoadedModelSignal::new(None));
            use_context_provider(|| Signal::new(PromptState::default()));

            rsx! {
                // Global app resources
                document::Link { rel: "icon", href: FAVICON }
                document::Link { rel: "stylesheet", href: MAIN_CSS }
                document::Title { "inferi chat" }

                if use_context::<LoadedModelSignal>().read().is_none() {
                    Home {}
                } else {
                    Chat {}
                }
            }
        }
        Some(Err(e)) => {
            rsx! {
                p {
                    strong {
                        { format!("WebGPU is not supported on this browser: {e}.") }
                    }
                }
                p {
                    "See ",
                    a {
                        href: "https://caniuse.com/webgpu",
                        " caniuse.com"
                    },
                    " for a list of compatible browsers."
                }
            }
        }
        _ => rsx! {},
    }
}
