use super::WhisperConfig;
use crate::context::LlmContext;
use crate::safetensor::SafeTensorExt;
use crate::tensor_cache::CachedTensor;
use khal::backend::{GpuBackend, GpuBackendError};
use khal::BufferUsages;
use nalgebra::DMatrix;
use safetensors::SafeTensors;
use std::sync::Arc;
use vortx::tensor::{Tensor, TensorBuilder};

struct Linear {
    weight: Tensor<f32>,
    bias: Option<Tensor<f32>>,
}

impl Linear {
    // TODO: are the size arguments really useful? Sounds like its already known from the
    //       tensor loading.
    pub fn new(loader: &ModelLoader<'_>, _a: usize, _b: usize) -> Result<Self, GpuBackendError> {
        Ok(Linear {
            weight: loader.tensor_f32("weight")?,
            bias: Some(loader.tensor_f32("bias")?),
        })
    }

    // TODO: are the size arguments really useful? Sounds like its already known from the
    //       tensor loading.
    pub fn without_bias(
        loader: &ModelLoader<'_>,
        _a: usize,
        _b: usize,
    ) -> Result<Self, GpuBackendError> {
        Ok(Linear {
            weight: loader.tensor_f32("weight")?,
            bias: None,
        })
    }

    pub fn forward(
        &self,
        ctx: &mut LlmContext<'_>,
        v: &Tensor<f32>,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        // FIXME: the matmul logic needs to be modified so it follows proper broadcasting rules and batch dimensions.
        //        For now, we just squeeze/cont/unsqueeze/cont so it works.
        let v_cont = ctx.contiguous(v.as_view().squeeze())?;
        let mut mv = ctx.matmul_ggml(&self.weight, &v_cont)?;
        let mut mv = mv.as_view_mut().unsqueeze(0);
        if let Some(bias) = &self.bias {
            ctx.add_assign(&mut mv, bias)?;
        }

        // FIXME: remove cont once matmul properly handles batch dimensions.
        ctx.contiguous(mv)
    }
}

struct LayerNorm {
    weight: Tensor<f32>,
    bias: Option<Tensor<f32>>,
    eps: f32,
}

impl LayerNorm {
    pub fn new(loader: &ModelLoader<'_>, _a: usize, eps: f32) -> Result<Self, GpuBackendError> {
        Ok(Self {
            weight: loader.tensor_f32("weight")?,
            bias: loader.tensor_f32("bias").ok(),
            eps,
        })
    }

    pub fn forward(
        &self,
        ctx: &mut LlmContext<'_>,
        x: &Tensor<f32>,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let mut n = ctx.layernorm(x, self.eps)?;
        ctx.mul_assign(&mut n, &self.weight)?;
        if let Some(bias) = &self.bias {
            ctx.add_assign(&mut n, bias)?;
        }

        Ok(n)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv1dConfig {
    pub padding: u32,
    pub stride: u32,
    pub dilation: u32,
    pub groups: u32,
}

impl Default for Conv1dConfig {
    fn default() -> Self {
        Self {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
        }
    }
}

struct Conv1d {
    weight: Tensor<f32>,
    bias: Option<Tensor<f32>>,
    config: Conv1dConfig,
}

impl Conv1d {
    pub fn new(
        loader: &ModelLoader<'_>,
        _in_channels: usize,
        _out_channels: usize,
        _kernel_size: usize,
        config: Conv1dConfig,
    ) -> Result<Self, GpuBackendError> {
        Ok(Self {
            weight: loader.tensor_f32("weight")?,
            bias: loader.tensor_f32("bias").ok(),
            config,
        })
    }

    pub fn forward(
        &self,
        ctx: &mut LlmContext<'_>,
        x: &Tensor<f32>,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let ws = self.weight.layout().size;
        let wi = x.layout().size;
        let k = ctx
            .contiguous(self.weight.reshape(&[ws[1], ws[2], ws[0]]))
            .unwrap();
        let i = ctx.contiguous(x.reshape(&[wi[1], wi[2], wi[0]])).unwrap();
        let im2col = ctx.im2col(
            &k,
            &i,
            self.config.stride,
            0,
            self.config.padding,
            0,
            self.config.dilation,
            0,
            false,
        )?;
        // println!("im2col: {:?}", im2col.layout());
        // let im2col_perm = ctx.contiguous(im2col.permute([0, 1, 2, 3]))?;
        // return Ok(im2col_perm);
        let ks = self.weight.layout().size;
        let mm_k = self.weight.reshape(&[ks[0], ks[1] * ks[2]]);
        let mm_col = im2col.as_view().squeeze().transpose_last_dims();
        let res = ctx.matmul(mm_k, mm_col)?;

        let res = if let Some(b) = &self.bias {
            ctx.add(res.as_view(), b.as_view().unsqueeze(1))?
        } else {
            res
        };

        let sr = res.layout().size;
        let res = ctx.contiguous(res.reshape(&[sr[2], sr[0], sr[1]]))?;
        Ok(res)
    }
}

pub struct Embedding {
    embeddings: Tensor<f32>,
    hidden_size: usize,
}

impl Embedding {
    pub fn new(
        loader: &ModelLoader<'_>,
        _in_size: usize,
        out_size: usize,
    ) -> Result<Self, GpuBackendError> {
        Ok(Self {
            embeddings: loader.tensor_f32("weight")?,
            hidden_size: out_size,
        })
    }

    pub fn forward(
        &self,
        ctx: &mut LlmContext<'_>,
        x: &Tensor<u32>,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let final_rank = x.rank() + 1;
        let mut final_dims = x.layout().size;
        final_dims[x.rank() as usize] = self.hidden_size as u32;
        let idx = x.reshape(&[x.len() as u32]);
        let idx_cont = ctx.contiguous(idx)?;
        let result = ctx.select(&self.embeddings, &idx_cont, 0)?;
        let reshape_array = [final_dims[0], final_dims[1], final_dims[2]];
        assert_eq!(reshape_array.len() as u32, final_rank);
        ctx.contiguous(result.reshape(&reshape_array)) // TODO: will be nicer to implement once reshape takes a &[u32] instead of an array.
    }
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L62
struct MultiHeadAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    n_head: usize,
    #[allow(clippy::type_complexity)]
    kv_cache: Option<(Arc<CachedTensor<f32>>, Arc<CachedTensor<f32>>)>,
}

impl MultiHeadAttention {
    fn new(
        loader: ModelLoader<'_>,
        n_state: usize,
        n_head: usize,
    ) -> Result<Self, GpuBackendError> {
        let query = Linear::new(&loader.var("q_proj"), n_state, n_state)?;
        let value = Linear::new(&loader.var("v_proj"), n_state, n_state)?;
        let key = Linear::without_bias(&loader.var("k_proj"), n_state, n_state)?;
        let out = Linear::new(&loader.var("out_proj"), n_state, n_state)?;
        Ok(Self {
            query,
            key,
            value,
            out,
            n_head,
            kv_cache: None,
        })
    }

    fn forward(
        &mut self,
        ctx: &mut LlmContext<'_>,
        x: &Tensor<f32>,
        xa: Option<&Tensor<f32>>,
        mask: Option<&Tensor<f32>>,
        flush_cache: bool,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        self.forward_dbg(ctx, x, xa, mask, flush_cache, false)
    }

    fn forward_dbg(
        &mut self,
        ctx: &mut LlmContext<'_>,
        x: &Tensor<f32>,
        xa: Option<&Tensor<f32>>,
        mask: Option<&Tensor<f32>>,
        flush_cache: bool,
        dbg: bool,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let q = self.query.forward(ctx, x)?;
        let (k, v) = match xa {
            None => {
                let k = self.key.forward(ctx, x)?;
                let v = self.value.forward(ctx, x)?;
                (Arc::new(k), Arc::new(v))
            }
            Some(x) => {
                if flush_cache {
                    self.kv_cache = None;
                }
                if let Some((k, v)) = &self.kv_cache {
                    (k.clone(), v.clone())
                } else {
                    let k = Arc::new(self.key.forward(ctx, x)?);
                    let v = Arc::new(self.value.forward(ctx, x)?);
                    self.kv_cache = Some((k.clone(), v.clone()));
                    (k, v)
                }
            }
        };
        let wv = self.qkv_attention_dbg(ctx, &q, &k, &v, mask, dbg)?;
        if dbg {
            return Ok(wv);
        }
        // println!("$$$$$ Before out: {:?}", wv.as_view().layout());
        let out = self.out.forward(ctx, &wv)?;
        Ok(out)
    }

    fn reshape_head(
        &self,
        ctx: &mut LlmContext<'_>,
        x: &Tensor<f32>,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let [n_batch, n_ctx, n_state, _] = x.layout().size;
        let target_dims = [
            n_batch,
            n_ctx,
            self.n_head as u32,
            n_state / self.n_head as u32,
        ];
        // TODO PERF: return just a view?
        ctx.contiguous(x.reshape(&target_dims).transpose(1, 2))
    }

    #[allow(dead_code)]
    fn qkv_attention(
        &self,
        ctx: &mut LlmContext<'_>,
        q: &Tensor<f32>,
        k: &Tensor<f32>,
        v: &Tensor<f32>,
        mask: Option<&Tensor<f32>>,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        self.qkv_attention_dbg(ctx, q, k, v, mask, false)
    }

    fn qkv_attention_dbg(
        &self,
        ctx: &mut LlmContext<'_>,
        q: &Tensor<f32>,
        k: &Tensor<f32>,
        v: &Tensor<f32>,
        mask: Option<&Tensor<f32>>,
        dbg: bool,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let [_, n_ctx, n_state, _] = q.layout().size;
        let scale = ((n_state / self.n_head as u32) as f64).powf(-0.25) as f32;

        let reshaped_q = self.reshape_head(ctx, q)?;
        let q = ctx.scale(&reshaped_q, scale)?;
        let reshaped_k = self.reshape_head(ctx, k)?;
        let k = ctx.scale(reshaped_k.transpose(2, 3), scale)?;
        let reshaped_v = self.reshape_head(ctx, v)?;
        let v = ctx.contiguous(&reshaped_v)?;
        let mut qk = ctx.matmul(&q, &k)?;

        if let Some(mask) = mask {
            // FIXME: add range indexing to avoid the double `narrow`.
            let mask = mask.narrow(0, 0, n_ctx).narrow(1, 0, n_ctx);
            ctx.add_assign(&mut qk, mask)?
        }

        ctx.softmax_rows(&mut qk)?;
        if dbg {
            println!(
                "scale: {}, n_state: {}, n_head: {}",
                scale, n_state, self.n_head
            );
            return Ok(qk);
        }
        let w = qk;
        let wv = ctx.matmul(&w, &v)?;
        let result = ctx.contiguous(wv.transpose(1, 2))?;
        // FIXME: replace the reshape by a `result.flatten_from(2);` And avoid the contiguous at the end.
        let result_shape = result.layout().size;
        let result_reshaped = result.reshape(&[
            result_shape[0],
            result_shape[1],
            result_shape[2] * result_shape[3],
        ]);
        ctx.contiguous(result_reshaped)
    }

    fn reset_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L111
struct ResidualAttentionBlock {
    attn: MultiHeadAttention,
    attn_ln: LayerNorm,
    cross_attn: Option<(MultiHeadAttention, LayerNorm)>,
    mlp_linear1: Linear,
    mlp_linear2: Linear,
    mlp_ln: LayerNorm,
}

impl ResidualAttentionBlock {
    fn new(
        loader: ModelLoader<'_>,
        _cfg: &WhisperConfig,
        n_state: usize,
        n_head: usize,
        ca: bool,
    ) -> Result<Self, GpuBackendError> {
        let attn = MultiHeadAttention::new(loader.var("self_attn"), n_state, n_head)?;
        let attn_ln = LayerNorm::new(&loader.var("self_attn_layer_norm"), n_state, 1.0e-5)?;
        let cross_attn = if ca {
            let cross_attn = MultiHeadAttention::new(loader.var("encoder_attn"), n_state, n_head)?;
            let cross_attn_ln =
                LayerNorm::new(&loader.var("encoder_attn_layer_norm"), n_state, 1.0e-5)?;
            Some((cross_attn, cross_attn_ln))
        } else {
            None
        };
        let n_mlp = n_state * 4;
        let mlp_linear1 = Linear::new(&loader.var("fc1"), n_state, n_mlp)?;
        let mlp_linear2 = Linear::new(&loader.var("fc2"), n_mlp, n_state)?;
        let mlp_ln = LayerNorm::new(&loader.var("final_layer_norm"), n_state, 1.0e-5)?;
        Ok(Self {
            attn,
            attn_ln,
            cross_attn,
            mlp_linear1,
            mlp_linear2,
            mlp_ln,
        })
    }

    fn forward(
        &mut self,
        ctx: &mut LlmContext<'_>,
        x: &Tensor<f32>,
        xa: Option<&Tensor<f32>>,
        mask: Option<&Tensor<f32>>,
        flush_kv_cache: bool,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        self.forward_dbg(ctx, x, xa, mask, flush_kv_cache, false)
    }

    fn forward_dbg(
        &mut self,
        ctx: &mut LlmContext<'_>,
        x: &Tensor<f32>,
        xa: Option<&Tensor<f32>>,
        mask: Option<&Tensor<f32>>,
        flush_kv_cache: bool,
        debug: bool,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let x_ln = self.attn_ln.forward(ctx, x)?;
        let attn = self
            .attn
            .forward_dbg(ctx, &x_ln, None, mask, flush_kv_cache, debug)?;

        if debug {
            return Ok(attn);
        }

        let mut x = ctx.add(x, &attn)?;

        if let Some((attn, ln)) = &mut self.cross_attn {
            let x_ln = ln.forward(ctx, &x)?;
            let x_attn = attn.forward(ctx, &x_ln, xa, None, flush_kv_cache)?;
            ctx.add_assign(&mut x, &x_attn)?;
        }

        let x_ln = self.mlp_ln.forward(ctx, &x)?;
        let x_len = self.mlp_linear1.forward(ctx, &x_ln)?;
        let x_gelu = ctx.gelu(&x_len)?;
        let mlp = self.mlp_linear2.forward(ctx, &x_gelu)?;
        ctx.add(&x, &mlp)
    }

    fn reset_kv_cache(&mut self) {
        self.attn.reset_kv_cache();
        if let Some((attn, _)) = &mut self.cross_attn {
            attn.reset_kv_cache();
        }
    }
}

fn sinusoids(
    backend: &GpuBackend,
    length: usize,
    channels: usize,
) -> Result<Tensor<f32>, GpuBackendError> {
    let max_timescale = 10000f32;
    let log_timescale_increment = max_timescale.ln() / (channels / 2 - 1) as f32;

    let scaled_time = DMatrix::from_fn(length, channels / 2, |l, c| {
        let cc = (c as f32 * (-log_timescale_increment)).exp();
        let ll = l as f32;
        ll * cc
    });
    #[allow(clippy::toplevel_ref_arg)]
    let sc = nalgebra::stack![scaled_time.map(|x| x.sin()), scaled_time.map(|x| x.cos())];
    let row_maj_sc = sc.transpose();
    TensorBuilder::matrix(sc.nrows() as u32, sc.ncols() as u32, BufferUsages::STORAGE)
        .build_init(backend, row_maj_sc.as_slice())

    // let inv_timescales: Vec<_> = (0..channels / 2)
    //     .map(|i| (i as f32 * (-log_timescale_increment)).exp())
    //     .collect();
    // // [1, channels / 2]
    // // 0.exp (1 * -log).exp() (2 * -log).exp() ... (m * -log).exp()
    // // 0.exp (1 * -log).exp() (2 * -log).exp() ... (m * -log).exp()
    // // 0.exp (1 * -log).exp() (2 * -log).exp() ... (m * -log).exp()
    // let inv_timescales = Tensor::new(inv_timescales.as_slice(), device)?.unsqueeze(0)?;
    // // [length, 1] =
    // // 0 0 ... 0
    // // 1 1 ... 1
    // // 2 2 ... 2
    // // n n ... n
    // let arange = Tensor::arange(0, length as u32, device)?
    //     .to_dtype(candle::DType::F32)?
    //     .unsqueeze(1)?;
    // let sh = (length, channels / 2);
    // let scaled_time = (arange.broadcast_as(sh)? * inv_timescales.broadcast_as(sh)?)?;
    // let sincos = Tensor::cat(&[scaled_time.sin()?, scaled_time.cos()?], 1)?;
    // Ok(sincos)
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L143
pub struct AudioEncoder {
    conv1: Conv1d,
    conv2: Conv1d,
    positional_embedding: Tensor<f32>,
    blocks: Vec<ResidualAttentionBlock>,
    ln_post: LayerNorm,
}

impl AudioEncoder {
    fn new(loader: ModelLoader<'_>, cfg: &WhisperConfig) -> Result<Self, GpuBackendError> {
        let n_state = cfg.d_model;
        let n_head = cfg.encoder_attention_heads;
        let n_ctx = cfg.max_source_positions;
        let cfg1 = Conv1dConfig {
            padding: 1,
            stride: 1,
            groups: 1,
            dilation: 1,
        };
        let cfg2 = Conv1dConfig {
            padding: 1,
            stride: 2,
            groups: 1,
            dilation: 1,
        };
        let conv1 = Conv1d::new(&loader.var("conv1"), cfg.num_mel_bins, n_state, 3, cfg1)?;
        let conv2 = Conv1d::new(&loader.var("conv2"), n_state, n_state, 3, cfg2)?;
        let positional_embedding = sinusoids(loader.backend, n_ctx, n_state)?;
        let blocks = (0..cfg.encoder_layers)
            .map(|i| {
                ResidualAttentionBlock::new(
                    loader.var(format!("layers.{i}")),
                    cfg,
                    n_state,
                    n_head,
                    false,
                )
            })
            .collect::<Result<Vec<_>, GpuBackendError>>()?;
        let ln_post = LayerNorm::new(&loader.var("layer_norm"), n_state, 1.0e-5)?;
        Ok(Self {
            conv1,
            conv2,
            positional_embedding,
            blocks,
            ln_post,
        })
    }

    pub async fn forward(
        &mut self,
        ctx: &mut LlmContext<'_>,
        x: &Tensor<f32>,
        flush_kv_cache: bool,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let x = {
            let conv = self.conv1.forward(ctx, x)?;
            ctx.gelu(conv.as_ref())?
        };

        let x = {
            let conv = self.conv2.forward(ctx, &x)?;
            ctx.gelu(conv.as_ref())?
        };

        let x = x.transpose(1, 2);

        let [_bsize, seq_len, _hidden, _] = x.layout().size;
        let positional_embedding = self.positional_embedding.narrow(0, 0, seq_len);

        let mut x = ctx.add(x, positional_embedding)?;

        for block in self.blocks.iter_mut() {
            x = block.forward(ctx, &x, None, None, flush_kv_cache)?;
        }
        let x = self.ln_post.forward(ctx, &x)?;

        Ok(x)
    }
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L176
pub struct TextDecoder {
    token_embedding: Embedding, // TODO
    positional_embedding: Tensor<f32>,
    blocks: Vec<ResidualAttentionBlock>,
    ln: LayerNorm,
    mask: Tensor<f32>,
}

impl TextDecoder {
    fn new(loader: ModelLoader<'_>, cfg: &WhisperConfig) -> Result<Self, GpuBackendError> {
        let n_state = cfg.d_model;
        let n_head = cfg.decoder_attention_heads;
        let n_ctx = cfg.max_target_positions;
        let token_embedding = Embedding::new(&loader.var("embed_tokens"), cfg.vocab_size, n_state)?;
        let positional_embedding = loader.tensor_f32("embed_positions.weight")?;
        let blocks = (0..cfg.decoder_layers)
            .map(|i| {
                ResidualAttentionBlock::new(
                    loader.var(format!("layers.{i}")),
                    cfg,
                    n_state,
                    n_head,
                    true,
                )
            })
            .collect::<Result<Vec<_>, GpuBackendError>>()?;
        let ln = LayerNorm::new(&loader.var("layer_norm"), n_state, 1.0e-5)?;
        let mask: Vec<_> = (0..n_ctx)
            .flat_map(|i| (0..n_ctx).map(move |j| if j > i { f32::NEG_INFINITY } else { 0f32 }))
            .collect();
        let mask = TensorBuilder::matrix(n_ctx as u32, n_ctx as u32, BufferUsages::STORAGE)
            .build_init(loader.backend, &mask)?;
        Ok(Self {
            token_embedding,
            positional_embedding,
            blocks,
            ln,
            mask,
        })
    }

    pub fn forward(
        &mut self,
        ctx: &mut LlmContext<'_>,
        x: &Tensor<u32>,
        xa: &Tensor<f32>,
        flush_kv_cache: bool,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let last = x.size(x.rank() as usize - 1);
        let token_embedding = self.token_embedding.forward(ctx, x)?;
        let positional_embedding = self.positional_embedding.narrow(0, 0, last);

        let mut x = ctx.add(&token_embedding, positional_embedding)?;

        for block in self.blocks.iter_mut() {
            let exit = false; // k == 0;
            x = block.forward_dbg(ctx, &x, Some(xa), Some(&self.mask), flush_kv_cache, exit)?;
            if exit {
                return Ok(x);
            }
        }

        self.ln.forward(ctx, &x)
    }

    pub fn final_linear(
        &self,
        ctx: &mut LlmContext<'_>,
        x: &Tensor<f32>,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        // let b_size = x.size(0);
        let w = &self.token_embedding.embeddings; // TOTO: is .broadcast_left(b_size)? needed (probably not since inferi auto-broadcasts);
        let wt = ctx.contiguous(w.as_view().transpose_last_dims())?;
        let logits = ctx.matmul(x, &wt)?;
        Ok(logits)
    }

    pub fn reset_kv_cache(&mut self) {
        for block in self.blocks.iter_mut() {
            block.reset_kv_cache();
        }
    }
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L221
pub struct Whisper {
    pub encoder: AudioEncoder,
    pub decoder: TextDecoder,
    pub config: WhisperConfig,
}

impl Whisper {
    pub fn new(loader: &ModelLoader<'_>, config: WhisperConfig) -> Result<Self, GpuBackendError> {
        let encoder = AudioEncoder::new(loader.var("model.encoder"), &config)?;
        let decoder = TextDecoder::new(loader.var("model.decoder"), &config)?;
        Ok(Self {
            encoder,
            decoder,
            config,
        })
    }

    pub fn reset_kv_cache(&mut self) {
        self.encoder
            .blocks
            .iter_mut()
            .for_each(|b| b.reset_kv_cache());
        self.decoder.reset_kv_cache();
    }
}

pub struct ModelLoader<'a> {
    backend: &'a GpuBackend,
    st: &'a SafeTensors<'a>,
    prefix: String,
}

impl Clone for ModelLoader<'_> {
    fn clone(&self) -> Self {
        Self {
            backend: self.backend,
            st: self.st,
            prefix: self.prefix.clone(),
        }
    }
}

impl<'a> ModelLoader<'a> {
    pub fn new(backend: &'a GpuBackend, st: &'a SafeTensors<'a>) -> Self {
        Self {
            backend,
            st,
            prefix: String::new(),
        }
    }

    pub fn var(&self, path: impl AsRef<str>) -> Self {
        let new_prefix = if self.prefix.is_empty() {
            path.as_ref().to_string()
        } else {
            format!("{}.{}", self.prefix, path.as_ref())
        };

        Self {
            backend: self.backend,
            st: self.st,
            prefix: new_prefix,
        }
    }

    pub fn tensor_f32(&self, path: &str) -> Result<Tensor<f32>, GpuBackendError> {
        let name = format!("{}.{}", self.prefix, path);
        // TODO: error propagation.
        let result = self
            .st
            .tensor(&name)
            .unwrap()
            .to_gpu_tensor_f32(self.backend)?;
        Ok(result)
    }
}
