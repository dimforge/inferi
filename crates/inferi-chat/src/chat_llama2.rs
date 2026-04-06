use crate::chat_template::ChatTemplate;
use crate::llm::AnyTokenizer;
use crate::prompt::{ChatEvent, Prompt};
use crate::sampler::{sample_next_token, SamplerParams};
use async_std::sync::RwLock;
use inferi::context::{LlmContext, LlmOps};
use inferi::gguf::Gguf;
use inferi::models::llama2::cpu::Llama2Config;
use inferi::models::llama2::{Llama2, Llama2State, Llama2Weights, LlamaModelType};
use inferi::re_exports::safetensors::SafeTensors;
use inferi::re_exports::vortx::shapes::TensorLayoutBuffers;
use inferi::tensor_cache::TensorCache;
use khal::backend::{Backend, GpuBackend};
use nalgebra::DVector;

pub struct ChatLlama2 {
    ops: LlmOps,
    transformer: Llama2,
    weights: Llama2Weights,
    tokenizer: AnyTokenizer,
    config: Llama2Config,
    state: RwLock<Llama2State>,
    shapes: RwLock<TensorLayoutBuffers>,
    tensor_cache: RwLock<TensorCache>,
}

impl ChatLlama2 {
    pub fn from_safetensors(
        backend: &GpuBackend,
        st: &SafeTensors,
        gguf: &Gguf,
    ) -> anyhow::Result<ChatLlama2> {
        Self::from_safetensors_with_model_type(backend, st, gguf, LlamaModelType::Llama)
    }

    pub fn from_gguf(backend: &GpuBackend, gguf: &Gguf) -> anyhow::Result<ChatLlama2> {
        Self::from_gguf_with_model_type(backend, gguf, LlamaModelType::Llama)
    }

    pub fn from_safetensors_with_model_type(
        backend: &GpuBackend,
        st: &SafeTensors,
        gguf: &Gguf, // TODO: remove this
        model_type: LlamaModelType,
    ) -> anyhow::Result<ChatLlama2> {
        let ops = LlmOps::new(backend)?;
        let transformer = Llama2::new(backend, model_type)?;
        let config = Llama2Config::from_gguf_with_model_type(gguf, model_type);
        println!("Config: {:#?}", config);
        let weights = Llama2Weights::from_safetensors(backend, &config, st)?;
        // let tokenizer = AnyTokenizer::from_gguf(gguf)?;
        // let gpt = Gpt2Tokenizer::from_gguf(gguf);
        let tokenizer = AnyTokenizer::from_hf("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")?;
        let state = Llama2State::new(backend, &config)?;

        Ok(Self {
            ops,
            transformer,
            weights,
            tokenizer,
            config,
            state: RwLock::new(state),
            shapes: RwLock::new(TensorLayoutBuffers::new(backend)),
            tensor_cache: RwLock::new(TensorCache::default()),
        })
    }

    pub fn from_gguf_with_model_type(
        backend: &GpuBackend,
        gguf: &Gguf,
        model_type: LlamaModelType,
    ) -> anyhow::Result<ChatLlama2> {
        let ops = LlmOps::new(backend)?;
        let transformer = Llama2::new(backend, model_type)?;
        let config = Llama2Config::from_gguf_with_model_type(gguf, model_type);
        let weights = Llama2Weights::from_gguf(backend, &config, gguf)?;
        let tokenizer = AnyTokenizer::from_gguf(gguf)?;
        let state = Llama2State::new(backend, &config)?;

        Ok(Self {
            ops,
            transformer,
            weights,
            tokenizer,
            config,
            state: RwLock::new(state),
            shapes: RwLock::new(TensorLayoutBuffers::new(backend)),
            tensor_cache: RwLock::new(TensorCache::default()),
        })
    }

    pub async fn forward(
        &self,
        backend: &GpuBackend,
        prompt: Prompt,
        sampler_params: SamplerParams,
        template: ChatTemplate,
        start_pos: usize,
        out: impl Fn(ChatEvent) -> anyhow::Result<()>,
    ) -> anyhow::Result<()> {
        log::info!("Original prompt:\n{}", prompt);

        let bos_str = self.tokenizer.bos_str();
        let eos_str = self.tokenizer.eos_str();
        // println!("eos_str: {}, bos_str: {}", eos_str, bos_str);
        let prompt_str = template.apply(&prompt, &bos_str, &eos_str);
        // println!("Forwarding prompt: '''{}'''", prompt_str);
        if out(ChatEvent::TemplatedPrompt(prompt_str.clone())).is_err() {
            return Ok(());
        }

        let (mut sampler, mut sampler_res) = sampler_params.sampler();
        let prompt_toks = self.tokenizer.encode(&prompt_str, false, false);
        // let prompt_toks =
        //     self.tokenizer
        //         .encode(&prompt_str, !prompt_str.starts_with(&bos_str), false);
        log::info!("Promp tokens: {:?}", prompt_toks);

        let prompt_toks_map: Vec<_> = prompt_toks
            .iter()
            .map(|tok| {
                let tok_str = self.tokenizer.decode(0, *tok);
                (*tok, tok_str)
            })
            .collect();
        if out(ChatEvent::PromptTokens(prompt_toks_map)).is_err() {
            return Ok(());
        }

        // Skip the first token in the tok/s timing since it is particularly slow due to gpu initialization.
        let timing_delay = 1;

        let mut token = prompt_toks[start_pos];
        let mut start = None;
        let mut logits = DVector::zeros(self.config.vocab_size);

        for pos in start_pos.. {
            if pos == start_pos + timing_delay {
                start = Some(web_time::Instant::now());
            }

            let t0 = web_time::Instant::now();
            self.forward_logits(backend, pos as u32, token as u32, &mut logits)
                .await?;
            let _elapsed = t0.elapsed().as_secs_f64();
            // println!("Logits time: {} (= {:.3} tok/s)", elapsed, 1.0 / elapsed);

            let _t0 = web_time::Instant::now();
            let next = sample_next_token(
                &mut sampler,
                &mut sampler_res,
                &mut logits,
                &prompt_toks,
                pos,
            );
            let token_string = self.tokenizer.decode(token, next);
            // println!("Sampling time: {}", t0.elapsed().as_secs_f64());
            token = next;

            if pos + 1 >= prompt_toks.len() {
                if token == self.tokenizer.eos() {
                    break;
                } else {
                    let (token_count, token_time) = if let Some(start) = &start {
                        (
                            pos - start_pos - timing_delay,
                            start.elapsed().as_secs_f64(),
                        )
                    } else {
                        (0, 0.0)
                    };

                    if out(ChatEvent::Token {
                        string: token_string,
                        next_pos: pos,
                        token_count,
                        token_time,
                    })
                    .is_err()
                    {
                        // Early-exit if an error was returned.
                        return Ok(());
                    }
                }
            }
        }

        Ok(())
    }

    async fn forward_logits(
        &self,
        backend: &GpuBackend,
        pos: u32,
        token: u32,
        out: &mut DVector<f32>,
    ) -> anyhow::Result<()> {
        let mut shapes = self.shapes.write().await;
        let mut tensor_cache = self.tensor_cache.write().await;
        let mut state = self.state.write().await;
        shapes.clear_tmp();
        // NOTE: tensor_cache.clear() removed - let the cache reuse buffers between tokens

        let (rope_config, rms_norm_config, attn_params) = self.config.derived_configs(pos);

        // Run the transformer.
        let _t0 = web_time::Instant::now();

        let mut encoder = backend.begin_encoding();
        backend.write_buffer(state.rope_config_mut().buffer_mut(), 0, &[rope_config])?;
        backend.write_buffer(
            state.rms_norm_config_mut().buffer_mut(),
            0,
            &[rms_norm_config],
        )?;
        backend.write_buffer(state.attn_params_mut().buffer_mut(), 0, &[attn_params])?;
        state
            .x
            .copy_from_view(&mut encoder, self.weights.token_embd.row(token))?;
        backend.submit(encoder)?;

        let _t0 = web_time::Instant::now();

        let mut ctxt = LlmContext {
            backend,
            shapes: &mut shapes,
            cache: &mut tensor_cache,
            pass: None,
            encoder: None,
            ops: &self.ops,
        };
        ctxt.begin_submission();
        self.transformer.launch(
            &mut ctxt,
            &mut state,
            &self.weights,
            &self.config,
            &attn_params,
            pos,
        )?;
        drop(ctxt.pass.take());

        let (logits, readback) = state.logits_and_readback_mut();
        readback.copy_from_view(ctxt.encoder.as_mut().unwrap(), logits)?;

        let _t0 = web_time::Instant::now();
        ctxt.submit();

        let _t0 = web_time::Instant::now();
        backend
            .read_buffer(state.logits_readback().buffer(), out.as_mut_slice())
            .await?;

        Ok(())
    }
}
