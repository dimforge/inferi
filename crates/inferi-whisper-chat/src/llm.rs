use crate::chat_llama2::ChatLlama2;
use crate::chat_template::ChatTemplate;
use crate::prompt::{ChatEvent, Prompt};
use crate::sampler::SamplerParams;
use inferi::gguf::{Gguf, GgufMetadataValue};
use inferi::models::llama2::LlamaModelType;
use inferi::models::tokenizers::{Gpt2Tokenizer, LlamaTokenizer};
use inferi::re_exports::tokenizers::Tokenizer;
use khal::backend::GpuBackend;

pub enum ChatLlm {
    Llama(ChatLlama2),
    Qwen(ChatLlama2),
}

impl ChatLlm {
    #[allow(dead_code)]
    pub fn model_name(&self) -> &'static str {
        match self {
            Self::Llama(_) => "llama",
            Self::Qwen(_) => "qwen2",
        }
    }
}

impl ChatLlm {
    pub async fn from_gguf(backend: &GpuBackend, gguf: &Gguf) -> anyhow::Result<Self> {
        let Some(GgufMetadataValue::String(name)) = gguf.metadata.get("general.architecture")
        else {
            anyhow::bail!("Unrecognized model")
        };

        if name.to_lowercase().contains("llama") {
            Ok(Self::Llama(ChatLlama2::from_gguf(backend, gguf)?))
        } else if name.to_lowercase().contains("qwen2") {
            Ok(Self::Qwen(ChatLlama2::from_gguf_with_model_type(
                backend,
                gguf,
                LlamaModelType::Qwen2,
            )?))
        } else {
            anyhow::bail!("Unrecognized model architecture: {}", name)
        }
    }

    pub async fn forward(
        &self,
        backend: &GpuBackend,
        prompt: Prompt,
        sampler_params: SamplerParams,
        chat_template: ChatTemplate,
        next_pos: usize,
        out: impl Fn(ChatEvent) -> anyhow::Result<()>,
    ) -> anyhow::Result<()> {
        match self {
            Self::Llama(llm) => {
                llm.forward(
                    backend,
                    prompt,
                    sampler_params,
                    chat_template,
                    next_pos,
                    out,
                )
                .await
            }
            Self::Qwen(llm) => {
                llm.forward(
                    backend,
                    prompt,
                    sampler_params,
                    chat_template,
                    next_pos,
                    out,
                )
                .await
            }
        }
    }
}

#[allow(clippy::large_enum_variant)]
pub enum AnyTokenizer {
    Llama(LlamaTokenizer),
    Gpt2(Gpt2Tokenizer),
    #[allow(dead_code)]
    Hf(Tokenizer),
}

impl AnyTokenizer {
    pub fn from_gguf(gguf: &Gguf) -> anyhow::Result<Self> {
        let tokenizer_type = gguf
            .metadata
            .get("tokenizer.ggml.model")
            .ok_or(anyhow::anyhow!("Missing tokenizer.ggml.model"))?
            .as_string();
        if tokenizer_type == "gpt2" {
            Ok(AnyTokenizer::Gpt2(Gpt2Tokenizer::from_gguf(gguf)))
        } else if tokenizer_type == "llama" {
            Ok(AnyTokenizer::Llama(LlamaTokenizer::from_gguf(gguf)))
        } else {
            anyhow::bail!("Unrecognized tokenizer type: {}", tokenizer_type)
        }
    }

    pub fn eos(&self) -> usize {
        match self {
            Self::Llama(t) => t.eos(),
            Self::Gpt2(t) => t.eos(),
            Self::Hf(_) => 151643, // FIXME
        }
    }

    #[allow(dead_code)]
    pub fn bos(&self) -> usize {
        match self {
            Self::Llama(t) => t.bos(),
            Self::Gpt2(t) => t.bos(),
            Self::Hf(_) => 151646, // FIXME
        }
    }

    pub fn bos_str(&self) -> String {
        match self {
            Self::Llama(t) => t.bos_str().to_string(),
            Self::Gpt2(t) => t.bos_str().to_string(),
            Self::Hf(t) => t.id_to_token(self.bos() as u32).unwrap(),
        }
    }

    pub fn eos_str(&self) -> String {
        match self {
            Self::Llama(t) => t.eos_str().to_string(),
            Self::Gpt2(t) => t.eos_str().to_string(),
            Self::Hf(t) => t.id_to_token(self.eos() as u32).unwrap(),
        }
    }

    pub fn decode(&self, prev_token: usize, token: usize) -> String {
        match self {
            Self::Llama(t) => t.decode(prev_token, token),
            Self::Gpt2(t) => t.decode(&[token as u32]),
            Self::Hf(t) => t.decode(&[token as u32], false).unwrap(),
        }
    }

    pub fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<usize> {
        match self {
            Self::Llama(t) => t.encode(text, bos, eos),
            Self::Gpt2(t) => t.encode(text),
            Self::Hf(t) => t
                .encode(text, true)
                .unwrap()
                .get_ids()
                .iter()
                .map(|id| *id as usize)
                .collect(),
        }
    }
}
