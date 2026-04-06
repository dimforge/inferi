use crate::chat_template::ChatTemplate;
use crate::llm::ChatLlm;
use crate::{GgufMetadata, GpuInstanceCtx, LoadedModel, LoadedModelSignal};
use dioxus::prelude::*;
use dioxus_markdown::Markdown;
use inferi::gguf::Gguf;
use inferi::models::segment_anything::SamGgmlFile;
use khal::backend::GpuBackend;
use rfd::FileHandle;
use std::sync::Arc;

const LOGO: Asset = asset!("/assets/inferi-logo.png");

#[cfg(target_arch = "wasm32")]
const DESCRIPTION: &str = "
The local LLM inference runs entirely **locally**. No data is sent to a server. The source code is available on GitHub for the [inferi](https://github.com/dimforge/inferi) library and this demo [inferi-chat](https://github.com/dimforge/inferi/tree/main/crates/inferi-chat). Supports llama/qwen models down to Q4 quantization.

----

Suggested models in the browser (limited to models < 1.5GB):
- [DeepSeek R1 Distill Qwen 1.5B Q4_0](https://huggingface.co/roleplaiapp/DeepSeek-R1-Distill-Qwen-1.5B-Q4_0-GGUF/blob/main/deepseek-r1-distill-qwen-1.5b-q4_0.gguf).
- [Tiny Llama 1.1B chat Q8_0](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/blob/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf).
";

#[cfg(not(target_arch = "wasm32"))]
const DESCRIPTION: &str = "
The local LLM inference runs entirely **locally**. No data is sent to a server. The source code is available on GitHub for the [inferi](https://github.com/dimforge/inferi) library and this demo [inferi-chat](https://github.com/dimforge/inferi/tree/main/crates/inferi-chat). Supports llama/qwen models down to Q4 quantization.

----

Suggested models on desktop (supports larger models than in the browser):
- [DeepSeek R1 Distill Qwen 1.5B Q4_0](https://huggingface.co/roleplaiapp/DeepSeek-R1-Distill-Qwen-1.5B-Q4_0-GGUF/blob/main/deepseek-r1-distill-qwen-1.5b-q4_0.gguf).
- [Tiny Llama 1.1B chat Q8_0](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/blob/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf).
- [DeepSeek R1 Distill Qwen 7B Q4_0](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF/blob/main/DeepSeek-R1-Distill-Qwen-7B-Q4_0.gguf).
- [Meta Llama 3 8B Instruct Q4_0](https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/blob/main/Meta-Llama-3-8B-Instruct.Q4_0.gguf).
";

#[component]
pub fn Home() -> Element {
    let gpu = use_context::<GpuInstanceCtx>().clone();
    let mut model = use_context::<LoadedModelSignal>();
    let mut model_file = use_signal(|| "".to_string());

    rsx! {
        div {
            id: "centered",
            a {
                style: "width:fit-content; margin-left:auto; margin-right:auto;",
                href: "https://github.com/dimforge/inferi",
                img { src: LOGO, id: "header" }
            }
            // div {
            //     id: "centered",
            //     h2 {
            //         "Local inference with Rust"
            //     }
            // }
            div {
                id: "header-text",
                Markdown {
                    src: DESCRIPTION,
                }
            }
        }
        if model_file.read().is_empty() {
            div {
                id: "gguf",
                a {
                    onclick: move |_event| {
                        let gpu = gpu.clone();
                        async move {
                            let task = rfd::AsyncFileDialog::new()
                                .add_filter("gguf model file", &["gguf", "bin"])
                                .pick_file();
                            let file = task.await;

                            if let Some(file) = file {
                                *model_file.write() = file.file_name();
                                *model.write() = Some(load_gguf(&gpu.backend, file).await);
                            }
                        }
                    },
                    "🤖 Open GGUF model"
                }
            }
        } else {
            div {
                id: "gguf",
                class: "gguf-loading",
                i {
                    { format!("⏳ {}", model_file.read()) }
                }
                {
                    #[cfg(target_arch = "wasm32")]
                    rsx! {
                        div {
                            class: "tiny-error",
                            "If the loading takes too long, check the JS console for errors. WASM might run out of memory."
                        }
                    }
                }
            }
        }
    }
}

async fn load_gguf(backend: &GpuBackend, file: FileHandle) -> LoadedModel {
    // let _ = gguf_snd.send(GgufLoadingProgress::ReadingFile).await;
    let bytes = file.read().await;
    // println!("Bytes read");
    // // let _ = gguf_snd.send(GgufLoadingProgress::ReadingTensors).await;
    //
    // let st_bytes = std::fs::read(
    //     "/Users/sebcrozet/work/trash/hf/DeepSeek-R1-Distill-Qwen-1.5B/model.safetensors",
    // )
    // .unwrap();
    // let st = SafeTensors::deserialize(&st_bytes).unwrap();
    //
    let gguf = if let Ok(sam) = SamGgmlFile::from_bytes(&bytes) {
        sam.into_gguf()
    } else {
        Gguf::from_bytes(&bytes).unwrap()
    };
    // let _ = gguf_snd
    //     .send(GgufLoadingProgress::PopulatingGpuResources)
    //     .await;
    drop(bytes); // Free up memory as soon as possible to save RAM (especially importanton WASM).

    // let a = Arc::new(ChatLlm::from_gguf(backend, &gguf).await.unwrap());
    // let b = Arc::new(
    //     ChatLlm::from_safetensors(backend, &st, &gguf)
    //         .await
    //         .unwrap(),
    // );

    LoadedModel {
        llm: Arc::new(ChatLlm::from_gguf(backend, &gguf).await.unwrap()),
        // llm: Arc::new(
        //     ChatLlm::from_safetensors(backend, &st, &gguf)
        //         .await
        //         .unwrap(),
        // ),
        sampler: Default::default(),
        template: ChatTemplate::from_gguf(&gguf),
        metadata: GgufMetadata {
            metadata: gguf.metadata_debug_strings().collect(),
            tensors: gguf.tensors_debug_strings().collect(),
        },
    }
}
