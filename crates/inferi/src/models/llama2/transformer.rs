use crate::context::LlmContext;
use crate::gguf::Gguf;
use crate::models::llama2::cpu::Llama2Config;
use crate::models::llama2::LlamaModelType;
use crate::ops::{
    BatchedMultiqueryAttention, BatchedMultiqueryAttentionParams, RmsNormConfig, RoPEConfig,
};
use crate::quantized_matrix::GpuQuantTensor;
use khal::backend::{GpuBackend, GpuBackendError};
use khal::{BufferUsages, Shader};
use nalgebra::DVector;
use safetensors::SafeTensors;
use vortx::tensor::Tensor;

pub struct Llama2State {
    /// Activation at current time stamp.
    pub x: Tensor<f32>,
    /// Activation at current time stamp, inside a residual branch.
    pub xb: Tensor<f32>,
    // DEBUG: useful for debugging the transformer.
    // pub xb_read: Tensor<f32>,
    /// Additional buffer for convenience.
    xb2: Tensor<f32>,
    /// Buffer for hidden dimension in the Feed-Forward net.
    hb: Tensor<f32>,
    /// Another buffer for hidden dimension in the Feed-Forward net.
    hb2: Tensor<f32>,
    /// Query.
    pub q: Tensor<f32>,
    // DEBUG: useful for debugging the transformer.
    // pub q_read: Tensor<f32>,
    /// Scores/attention values.
    att: Tensor<f32>,
    /// Output logits.
    logits: Tensor<f32>,
    logits_readback: Tensor<f32>,
    // KV cache. Each Vec contains `layer` elements.
    key_cache: Vec<Tensor<f32>>,
    value_cache: Vec<Tensor<f32>>,
    rope_config: Tensor<RoPEConfig>,
    rms_norm_config: Tensor<RmsNormConfig>,
    attn_params: Tensor<BatchedMultiqueryAttentionParams>,
}

impl Llama2State {
    pub fn new(backend: &GpuBackend, config: &Llama2Config) -> Result<Self, GpuBackendError> {
        let kv_dim = (config.hidden_size * config.num_key_value_heads) / config.num_attention_heads;
        const STORAGE: BufferUsages = BufferUsages::STORAGE;
        const UNIFORM: BufferUsages = BufferUsages::UNIFORM;

        let (rope_config, rms_norm_config, attn_params) = config.derived_configs(0);

        Ok(Self {
            x: Tensor::vector_uninit(
                backend,
                config.hidden_size as u32,
                STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            )?,
            xb: Tensor::vector_uninit(
                backend,
                config.hidden_size as u32,
                STORAGE | BufferUsages::COPY_SRC,
            )?,
            // DEBUG: useful for debugging the transformer.
            // xb_read: Tensor::vector_uninit(
            //     backend,
            //     config.dim as u32,
            //     BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            // )?,
            xb2: Tensor::vector_uninit(backend, config.hidden_size as u32, STORAGE)?,
            hb: Tensor::vector_uninit(backend, config.intermediate_size as u32, STORAGE)?,
            hb2: Tensor::vector_uninit(backend, config.intermediate_size as u32, STORAGE)?,
            q: Tensor::vector_uninit(
                backend,
                config.hidden_size as u32,
                STORAGE | BufferUsages::COPY_SRC,
            )?,
            // DEBUG: useful for debugging the transformer.
            // q_read: Tensor::vector_uninit(
            //     backend,
            //     config.dim as u32,
            //     BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            // )?,
            // TODO: for these two, the `kv_dim` doesn't match the dimension in the field's comment.
            key_cache: (0..config.num_hidden_layers)
                .map(|_| {
                    Tensor::matrix_uninit(
                        backend,
                        config.max_position_embeddings as u32,
                        kv_dim as u32,
                        STORAGE,
                    )
                })
                .collect::<Result<Vec<_>, GpuBackendError>>()?,
            value_cache: (0..config.num_hidden_layers)
                .map(|_| {
                    Tensor::matrix_uninit(
                        backend,
                        config.max_position_embeddings as u32,
                        kv_dim as u32,
                        STORAGE,
                    )
                })
                .collect::<Result<Vec<_>, GpuBackendError>>()?,
            att: Tensor::matrix_uninit(
                backend,
                config.max_position_embeddings as u32,
                config.num_attention_heads as u32,
                STORAGE,
            )?,
            logits: Tensor::vector_uninit(
                backend,
                config.vocab_size as u32,
                STORAGE | BufferUsages::COPY_SRC,
            )?,
            logits_readback: Tensor::vector_uninit(
                backend,
                config.vocab_size as u32,
                BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            )?,
            rope_config: Tensor::scalar(
                backend,
                rope_config,
                STORAGE | UNIFORM | BufferUsages::COPY_DST,
            )?,
            rms_norm_config: Tensor::scalar(
                backend,
                rms_norm_config,
                STORAGE | UNIFORM | BufferUsages::COPY_DST,
            )?,
            attn_params: Tensor::scalar(
                backend,
                attn_params,
                STORAGE | UNIFORM | BufferUsages::COPY_DST,
            )?,
        })
    }

    pub fn rope_config(&self) -> &Tensor<RoPEConfig> {
        &self.rope_config
    }

    pub fn rope_config_mut(&mut self) -> &mut Tensor<RoPEConfig> {
        &mut self.rope_config
    }

    pub fn rms_norm_config(&self) -> &Tensor<RmsNormConfig> {
        &self.rms_norm_config
    }

    pub fn rms_norm_config_mut(&mut self) -> &mut Tensor<RmsNormConfig> {
        &mut self.rms_norm_config
    }

    pub fn attn_params(&self) -> &Tensor<BatchedMultiqueryAttentionParams> {
        &self.attn_params
    }

    pub fn attn_params_mut(&mut self) -> &mut Tensor<BatchedMultiqueryAttentionParams> {
        &mut self.attn_params
    }

    pub fn logits(&self) -> &Tensor<f32> {
        &self.logits
    }

    pub fn logits_readback(&self) -> &Tensor<f32> {
        &self.logits_readback
    }

    pub fn logits_and_readback_mut(&mut self) -> (&Tensor<f32>, &mut Tensor<f32>) {
        (&self.logits, &mut self.logits_readback)
    }
}

pub struct Llama2LayerWeights {
    pub attn_norm: Tensor<f32>,
    pub attn_k: GpuQuantTensor,
    pub attn_q: GpuQuantTensor,
    pub attn_v: GpuQuantTensor,
    pub attn_k_bias: Option<Tensor<f32>>,
    pub attn_q_bias: Option<Tensor<f32>>,
    pub attn_v_bias: Option<Tensor<f32>>,
    pub ffn_down: GpuQuantTensor,
    pub ffn_gate: GpuQuantTensor,
    pub ffn_norm: Tensor<f32>,
    pub ffn_up: GpuQuantTensor,
    pub attn_output: GpuQuantTensor,
}

pub struct Llama2Weights {
    pub layers: Vec<Llama2LayerWeights>,
    pub token_embd: Tensor<f32>,
    pub output: GpuQuantTensor,
    pub output_norm: Tensor<f32>,
}

impl Llama2Weights {
    pub fn from_safetensors(
        backend: &GpuBackend,
        config: &Llama2Config,
        st: &SafeTensors,
    ) -> Result<Self, GpuBackendError> {
        use crate::safetensor::SafeTensorExt;

        // TODO: try out with https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/tree/main
        let usage = BufferUsages::STORAGE;
        let mut layers = vec![];

        for i_layer in 0..config.num_hidden_layers {
            log::info!("Loop {}/{}", i_layer, config.num_hidden_layers);
            let attn_q = format!("model.layers.{}.self_attn.q_proj.weight", i_layer);
            let attn_k = format!("model.layers.{}.self_attn.k_proj.weight", i_layer);
            let attn_v = format!("model.layers.{}.self_attn.v_proj.weight", i_layer);
            let attn_q_bias = format!("model.layers.{}.self_attn.q_proj.bias", i_layer);
            let attn_k_bias = format!("model.layers.{}.self_attn.k_proj.bias", i_layer);
            let attn_v_bias = format!("model.layers.{}.self_attn.v_proj.bias", i_layer);
            let attn_output = format!("model.layers.{}.self_attn.o_proj.weight", i_layer);
            let ffn_down = format!("model.layers.{}.mlp.down_proj.weight", i_layer);
            let ffn_gate = format!("model.layers.{}.mlp.gate_proj.weight", i_layer);
            let ffn_up = format!("model.layers.{}.mlp.up_proj.weight", i_layer);
            let ffn_norm = format!("model.layers.{}.post_attention_layernorm.weight", i_layer);
            let attn_norm = format!("model.layers.{}.input_layernorm.weight", i_layer);

            let attn_q = st.tensor(&attn_q).unwrap().to_gpu_tensor(backend)?;
            let attn_k = st.tensor(&attn_k).unwrap().to_gpu_tensor(backend)?;
            let attn_v = st.tensor(&attn_v).unwrap().to_gpu_tensor(backend)?;

            let attn_q_bias = st
                .tensor(&attn_q_bias)
                .ok()
                .map(|t| t.to_gpu_tensor_f32(backend))
                .transpose()?;
            let attn_k_bias = st
                .tensor(&attn_k_bias)
                .ok()
                .map(|t| t.to_gpu_tensor_f32(backend))
                .transpose()?;
            let attn_v_bias = st
                .tensor(&attn_v_bias)
                .ok()
                .map(|t| t.to_gpu_tensor_f32(backend))
                .transpose()?;
            let attn_output = st.tensor(&attn_output).unwrap().to_gpu_tensor(backend)?;
            let ffn_down = st.tensor(&ffn_down).unwrap().to_gpu_tensor(backend)?;
            let ffn_gate = st.tensor(&ffn_gate).unwrap().to_gpu_tensor(backend)?;
            let ffn_up = st.tensor(&ffn_up).unwrap().to_gpu_tensor(backend)?;

            let ffn_norm = st.tensor(&ffn_norm).unwrap().to_gpu_tensor_f32(backend)?;
            let attn_norm = st.tensor(&attn_norm).unwrap().to_gpu_tensor_f32(backend)?;

            layers.push(Llama2LayerWeights {
                attn_k,
                attn_norm,
                attn_q,
                attn_v,
                attn_k_bias,
                attn_q_bias,
                attn_v_bias,
                ffn_down,
                ffn_gate,
                ffn_norm,
                ffn_up,
                attn_output,
            });
        }

        log::info!("Loop done");
        let token_embd_name = "model.embed_tokens.weight";
        let output = "lm_head.weight";
        let output_norm = "model.norm.weight"; // TODO: is this correct?

        // TODO: keep the token embeddings in quantized form
        let token_embd = st
            .tensor(token_embd_name)
            .unwrap()
            .to_gpu_tensor_f32_with_usage(backend, usage | BufferUsages::COPY_SRC, false)?;

        let output = if let Ok(v) = st.tensor(output) {
            v.to_gpu_tensor(backend)?
        } else {
            st.tensor(token_embd_name).unwrap().to_gpu_tensor(backend)?
        };
        let output_norm = st.tensor(output_norm).unwrap().to_gpu_tensor_f32(backend)?;

        Ok(Self {
            layers,
            token_embd,
            output,
            output_norm,
        })
    }

    pub fn from_gguf(
        backend: &GpuBackend,
        config: &Llama2Config,
        gguf: &Gguf,
    ) -> Result<Self, GpuBackendError> {
        let usage = BufferUsages::STORAGE;
        let mut layers = vec![];

        for i_layer in 0..config.num_hidden_layers {
            log::info!("Loop {}/{}", i_layer, config.num_hidden_layers);
            let attn_q = format!("blk.{}.attn_q.weight", i_layer);
            let attn_k = format!("blk.{}.attn_k.weight", i_layer);
            let attn_v = format!("blk.{}.attn_v.weight", i_layer);
            let attn_q_bias = format!("blk.{}.attn_q.bias", i_layer);
            let attn_k_bias = format!("blk.{}.attn_k.bias", i_layer);
            let attn_v_bias = format!("blk.{}.attn_v.bias", i_layer);
            let attn_output = format!("blk.{}.attn_output.weight", i_layer);
            let ffn_down = format!("blk.{}.ffn_down.weight", i_layer);
            let ffn_gate = format!("blk.{}.ffn_gate.weight", i_layer);
            let ffn_up = format!("blk.{}.ffn_up.weight", i_layer);
            let ffn_norm = format!("blk.{}.ffn_norm.weight", i_layer);
            let attn_norm = format!("blk.{}.attn_norm.weight", i_layer);

            let attn_q = gguf.tensors[&attn_q].to_gpu_quant(backend)?.unwrap();
            let attn_k = gguf.tensors[&attn_k].to_gpu_quant(backend)?.unwrap();
            let attn_v = gguf.tensors[&attn_v].to_gpu_quant(backend)?.unwrap();

            let attn_q_bias = gguf
                .tensors
                .get(&attn_q_bias)
                .map(|t| Tensor::vector(backend, t.data().as_f32().unwrap(), usage))
                .transpose()?;
            let attn_k_bias = gguf
                .tensors
                .get(&attn_k_bias)
                .map(|t| Tensor::vector(backend, t.data().as_f32().unwrap(), usage))
                .transpose()?;
            let attn_v_bias = gguf
                .tensors
                .get(&attn_v_bias)
                .map(|t| Tensor::vector(backend, t.data().as_f32().unwrap(), usage))
                .transpose()?;
            let attn_output = gguf.tensors[&attn_output].to_gpu_quant(backend)?.unwrap();
            let ffn_down = gguf.tensors[&ffn_down].to_gpu_quant(backend)?.unwrap();
            let ffn_gate = gguf.tensors[&ffn_gate].to_gpu_quant(backend)?.unwrap();

            let ffn_up = gguf.tensors[&ffn_up].to_gpu_quant(backend)?.unwrap();

            let ffn_norm = gguf.tensors[&ffn_norm].data().as_f32().unwrap();
            let attn_norm = gguf.tensors[&attn_norm].data().as_f32().unwrap();

            layers.push(Llama2LayerWeights {
                attn_k,
                attn_norm: Tensor::vector(backend, attn_norm, usage)?,
                attn_q,
                attn_v,
                attn_k_bias,
                attn_q_bias,
                attn_v_bias,
                ffn_down,
                ffn_gate,
                ffn_norm: Tensor::vector(backend, ffn_norm, usage)?,
                ffn_up,
                attn_output,
            });
        }

        log::info!("Loop done");
        let token_embd_name = "token_embd.weight";
        let output = "output.weight";
        let output_norm = "output_norm.weight";

        // TODO: keep the token embeddings in quantized form
        let token_embd = &gguf.tensors[token_embd_name].data().dequantize().unwrap();
        let token_embd = Tensor::matrix(
            backend,
            config.vocab_size as u32,
            config.hidden_size as u32,
            token_embd,
            usage | BufferUsages::COPY_SRC,
        )?;

        let output = if let Some(v) = gguf.tensors.get(output) {
            v.to_gpu_quant(backend)?.unwrap()
        } else {
            gguf.tensors[token_embd_name]
                .to_gpu_quant(backend)?
                .unwrap()
        };
        let output_norm = gguf.tensors[output_norm].data().as_f32().unwrap();
        let output_norm = DVector::from_row_slice(output_norm);
        let output_norm = Tensor::vector(backend, &output_norm, usage);

        Ok(Self {
            layers,
            token_embd,
            output,
            output_norm: output_norm?,
        })
    }
}

pub struct Llama2 {
    model_type: LlamaModelType,
    attn: BatchedMultiqueryAttention,
}

impl Llama2 {
    pub fn new(backend: &GpuBackend, model_type: LlamaModelType) -> Result<Self, GpuBackendError> {
        Ok(Self {
            model_type,
            attn: BatchedMultiqueryAttention::from_backend(backend)?,
        })
    }

    pub fn launch(
        &self,
        ctxt: &mut LlmContext<'_>,
        state: &mut Llama2State,
        weights: &Llama2Weights,
        config: &Llama2Config,
        attn_params: &BatchedMultiqueryAttentionParams,
        pos: u32,
    ) -> Result<(), GpuBackendError> {
        for l in 0..config.num_hidden_layers {
            let wl = &weights.layers[l];
            ctxt.rms_norm_assign(
                &state.rms_norm_config,
                &mut state.xb,
                &state.x,
                &wl.attn_norm,
            )?;

            let mut k_cache = state.key_cache[l].row_mut(pos).squeeze();
            let mut v_cache = state.value_cache[l].row_mut(pos).squeeze();

            ctxt.matmul_quant_assign(&mut state.q, &wl.attn_q, &state.xb)?;
            ctxt.matmul_quant_assign(&mut k_cache, &wl.attn_k, &state.xb)?;
            ctxt.matmul_quant_assign(&mut v_cache, &wl.attn_v, &state.xb)?;

            if let Some(q_bias) = &wl.attn_q_bias {
                ctxt.add_assign(&mut state.q, q_bias)?;
            }
            if let Some(k_bias) = &wl.attn_k_bias {
                ctxt.add_assign(&mut k_cache, k_bias)?;
            }
            if let Some(v_bias) = &wl.attn_v_bias {
                ctxt.add_assign(&mut v_cache, v_bias)?;
            }

            let rope_variant = self.model_type.rope_variant();
            ctxt.rope(rope_variant, &state.rope_config, &mut state.q, k_cache)?;

            // Start attention.
            self.dispatch_attn(ctxt, state, l, attn_params)?;

            ctxt.matmul_quant_assign(&mut state.xb2, &wl.attn_output, &state.xb)?;
            // End attention.

            ctxt.add_assign(&mut state.x, &state.xb2)?;
            ctxt.rms_norm_assign(
                &state.rms_norm_config,
                &mut state.xb,
                &state.x,
                &wl.ffn_norm,
            )?;

            // Start ffn_silu
            ctxt.matmul_quant_assign(&mut state.hb, &wl.ffn_gate, &state.xb)?;
            ctxt.matmul_quant_assign(&mut state.hb2, &wl.ffn_up, &state.xb)?;
            ctxt.silu(&mut state.hb, &state.hb2)?;
            ctxt.matmul_quant_assign(&mut state.xb2, &wl.ffn_down, &state.hb)?;
            // End ffn_silu

            ctxt.add_assign(&mut state.x, &state.xb2)?;
        }

        ctxt.rms_norm_assign(
            &state.rms_norm_config,
            &mut state.xb,
            &state.x,
            &weights.output_norm,
        )?;

        ctxt.matmul_quant_assign(&mut state.logits, &weights.output, &state.xb)?;

        Ok(())
    }

    fn dispatch_attn(
        &self,
        ctxt: &mut LlmContext<'_>,
        state: &mut Llama2State,
        layer: usize,
        attn_params: &BatchedMultiqueryAttentionParams,
    ) -> Result<(), GpuBackendError> {
        self.attn.launch(
            ctxt,
            attn_params,
            &state.attn_params,
            &state.q,
            &state.key_cache[layer],
            &state.value_cache[layer],
            &mut state.att,
            &mut state.xb,
        )
    }
}
