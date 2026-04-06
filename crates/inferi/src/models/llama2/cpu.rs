//! The CPU version of the llama2 transformer.

use crate::gguf::Gguf;
use crate::models::llama2::LlamaModelType;
use crate::ops::{BatchedMultiqueryAttentionParams, RmsNormConfig, RoPEConfig};
use nalgebra::{
    vector, DMatrix, DVector, DVectorViewMut, Dyn, OMatrix, OVector, Rotation2, Storage,
    StorageMut, Vector,
};
use std::ffi::c_int;

type Dim = Dyn;
type HiddenDim = Dyn;
type NumHeads = Dyn;
type SeqLen = Dyn;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RawConfig {
    /// The transformer dimension.
    /// In particular, this is the size of an embedding.
    dim: c_int,
    /// Number of dimension of the feed-forward neural net.
    hidden_dim: c_int,
    /// Number of layers.
    n_layers: c_int,
    /// Number of query heads.
    n_q_heads: c_int,
    /// Number of key/value heads (can be < than `n_q_heads` because of multiquery).
    /// See <https://youtu.be/Mn_9W1nCFLo?si=UnkLuzaHlX8JKyjl&t=3808> (Grouped-query diagram).
    n_kv_heads: c_int,
    /// Vocabulary size, usually 256 (byte -level).
    vocab_size: c_int,
    /// Max sequence length.
    seq_len: c_int,
}

/*
 * Important note: the original code (like most of the LLM literature) assumes row-major matrices
 * with left-multiplication (vector * Matrix).
 * nalgebra uses column-major with right-multiplication (Matrix * vector). So in the end the data layout still match,
 * we just have to swap al the matrix dimensions, access columns instead of rows (and vice versa),
 * and replace left-multiplication by right-multiplication.
 */
#[derive(Copy, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Llama2Config {
    /// The transformer dimension.
    pub hidden_size: usize,
    /// Number of dimension of the feed-forward neural net.
    pub intermediate_size: usize,
    /// Number of layers.
    pub num_hidden_layers: usize,
    /// Number of query heads.
    pub num_attention_heads: usize,
    /// Number of key/value heads (can be < than `n_q_heads` because of multiquery).
    /// See <https://youtu.be/Mn_9W1nCFLo?si=UnkLuzaHlX8JKyjl&t=3808> (Grouped-query diagram).
    pub num_key_value_heads: usize,
    /// Vocabulary size, usually 256 (byte -level).
    pub vocab_size: usize,
    /// Max sequence length.
    pub max_position_embeddings: usize,
    /// The base frequency for Rotary Positional Encoding.
    pub rope_theta: f32,
    /// Nudge factor in the rms-norm kernel.
    pub rms_norm_eps: f32,
}

impl Llama2Config {
    pub fn read(bytes: &[u8]) -> Self {
        let elts: &[RawConfig] = bytemuck::cast_slice(&bytes[..std::mem::size_of::<RawConfig>()]);
        elts[0].into()
    }

    pub fn from_gguf(gguf: &Gguf) -> Self {
        Llama2Config::from_gguf_with_model_type(gguf, LlamaModelType::Llama)
    }

    pub fn from_gguf_with_model_type(gguf: &Gguf, model_type: LlamaModelType) -> Self {
        let model_name = model_type.gguf_model_name();
        let dim = format!("{model_name}.embedding_length");
        let hidden_dim = format!("{model_name}.feed_forward_length");
        let n_layers = format!("{model_name}.block_count");
        let n_q_heads = format!("{model_name}.attention.head_count");
        let n_kv_heads = format!("{model_name}.attention.head_count_kv");
        let seq_len = format!("{model_name}.context_length");
        let base_freq = format!("{model_name}.rope.freq_base");
        let rms_norm_eps = format!("{model_name}.attention.layer_norm_rms_epsilon");

        Self {
            hidden_size: gguf.metadata[&dim].unwrap_u32() as usize,
            intermediate_size: gguf.metadata[&hidden_dim].unwrap_u32() as usize,
            num_hidden_layers: gguf.metadata[&n_layers].unwrap_u32() as usize,
            num_attention_heads: gguf.metadata[&n_q_heads].unwrap_u32() as usize,
            num_key_value_heads: gguf.metadata[&n_kv_heads].unwrap_u32() as usize,
            vocab_size: gguf.metadata["tokenizer.ggml.tokens"].unwrap_array_len(),
            max_position_embeddings: gguf.metadata[&seq_len].unwrap_u32() as usize,
            rope_theta: gguf
                .metadata
                .get(&base_freq)
                .map(|x| x.as_f32())
                .unwrap_or(10000.0),
            rms_norm_eps: gguf
                .metadata
                .get(&rms_norm_eps)
                .map(|x| x.as_f32())
                .unwrap_or(1.0e-6),
        }
    }

    pub fn derived_configs(
        &self,
        token_pos: u32,
    ) -> (RoPEConfig, RmsNormConfig, BatchedMultiqueryAttentionParams) {
        let dim = self.hidden_size;
        let kv_dim =
            ((self.hidden_size * self.num_key_value_heads) / self.num_attention_heads) as u32;
        let head_size = (dim / self.num_attention_heads) as u32;
        let kv_mul = (self.num_attention_heads / self.num_key_value_heads) as u32;

        let rope_config = RoPEConfig {
            head_size,
            kv_dim,
            pos: token_pos,
            base_freq: self.rope_theta,
        };

        let rms_norm_config = RmsNormConfig {
            nudge_factor: self.rms_norm_eps,
        };

        let attn_params = BatchedMultiqueryAttentionParams {
            seq_len: self.max_position_embeddings as u32,
            kv_dim,
            kv_mul,
            n_heads: self.num_attention_heads as u32,
            head_size,
            pos: token_pos,
        };

        (rope_config, rms_norm_config, attn_params)
    }
}

impl From<RawConfig> for Llama2Config {
    fn from(c: RawConfig) -> Self {
        Self {
            hidden_size: c.dim as usize,
            intermediate_size: c.hidden_dim as usize,
            num_hidden_layers: c.n_layers as usize,
            num_attention_heads: c.n_q_heads as usize,
            num_key_value_heads: c.n_kv_heads as usize,
            vocab_size: c.vocab_size.unsigned_abs() as usize,
            max_position_embeddings: c.seq_len as usize,
            rope_theta: 10000.0,
            rms_norm_eps: 1.0e-6,
        }
    }
}

pub struct TransformerLayerWeights {
    pub attn_k: DMatrix<f32>,
    pub attn_norm: DVector<f32>,
    pub attn_q: DMatrix<f32>,
    pub attn_v: DMatrix<f32>,
    pub ffn_down: DMatrix<f32>,
    pub ffn_gate: DMatrix<f32>,
    pub ffn_norm: DVector<f32>,
    pub ffn_up: DMatrix<f32>,
    pub attn_output: DMatrix<f32>,
}

pub struct TransformerWeights {
    pub layers: Vec<TransformerLayerWeights>,
    pub token_embd: DMatrix<f32>,
    pub output: DMatrix<f32>,
    pub output_norm: DVector<f32>,
}

impl TransformerWeights {
    pub fn from_gguf(config: &Llama2Config, gguf: &Gguf) -> Self {
        let head_size = config.hidden_size / config.num_attention_heads;
        let num_kv_heads_times_head_size = config.num_key_value_heads * head_size;

        let mut layers = vec![];

        for i_layer in 0..config.num_hidden_layers {
            log::info!("Loop {}/{}", i_layer, config.num_hidden_layers);
            let attn_q = format!("blk.{}.attn_q.weight", i_layer);
            let attn_k = format!("blk.{}.attn_k.weight", i_layer);
            let attn_v = format!("blk.{}.attn_v.weight", i_layer);
            let attn_output = format!("blk.{}.attn_output.weight", i_layer);
            let ffn_down = format!("blk.{}.ffn_down.weight", i_layer);
            let ffn_gate = format!("blk.{}.ffn_gate.weight", i_layer);
            let ffn_up = format!("blk.{}.ffn_up.weight", i_layer);
            let ffn_norm = format!("blk.{}.ffn_norm.weight", i_layer);
            let attn_norm = format!("blk.{}.attn_norm.weight", i_layer);

            let attn_q = &gguf.tensors[&attn_q].data().dequantize().unwrap();
            let attn_k = &gguf.tensors[&attn_k].data().dequantize().unwrap();
            let attn_v = &gguf.tensors[&attn_v].data().dequantize().unwrap();
            let attn_output = &gguf.tensors[&attn_output].data().dequantize().unwrap();
            let ffn_down = &gguf.tensors[&ffn_down].data().dequantize().unwrap();
            let ffn_gate = &gguf.tensors[&ffn_gate].data().dequantize().unwrap();
            let ffn_up = &gguf.tensors[&ffn_up].data().dequantize().unwrap();
            let ffn_norm = gguf.tensors[&ffn_norm].data().as_f32().unwrap();
            let attn_norm = gguf.tensors[&attn_norm].data().as_f32().unwrap();

            let ffn_norm = DVector::from_row_slice(ffn_norm);
            let attn_norm = DVector::from_row_slice(attn_norm);

            let attn_q = DMatrix::from_row_slice(config.hidden_size, config.hidden_size, attn_q);
            let attn_k =
                DMatrix::from_row_slice(num_kv_heads_times_head_size, config.hidden_size, attn_k);
            let attn_v =
                DMatrix::from_row_slice(num_kv_heads_times_head_size, config.hidden_size, attn_v);
            let attn_output =
                DMatrix::from_row_slice(config.hidden_size, config.hidden_size, attn_output);
            let ffn_down =
                DMatrix::from_row_slice(config.hidden_size, config.intermediate_size, ffn_down);
            let ffn_gate =
                DMatrix::from_row_slice(config.intermediate_size, config.hidden_size, ffn_gate);
            let ffn_up =
                DMatrix::from_row_slice(config.intermediate_size, config.hidden_size, ffn_up);

            layers.push(TransformerLayerWeights {
                attn_q,
                attn_k,
                attn_v,
                attn_output,
                ffn_down,
                ffn_gate,
                ffn_up,
                ffn_norm,
                attn_norm,
            });
        }

        log::info!("Loop done");
        let token_embd = "token_embd.weight";
        let output = "output.weight";
        let output_norm = "output_norm.weight";

        let token_embd = &gguf.tensors[token_embd].data().dequantize().unwrap();
        let output = gguf
            .tensors
            .get(output)
            .map(|v| v.data().dequantize().unwrap());
        let output_norm = gguf.tensors[output_norm].data().as_f32().unwrap();

        let token_embd =
            DMatrix::from_column_slice(config.hidden_size, config.vocab_size, token_embd);
        let output = output
            .map(|data| DMatrix::from_row_slice(config.vocab_size, config.hidden_size, &data))
            .unwrap_or_else(|| token_embd.transpose());
        let output_norm = DVector::from_row_slice(output_norm);

        Self {
            layers,
            token_embd,
            output,
            output_norm,
        }
    }
}

struct RunState {
    // Current wave of activations.
    /// Activation at current time stamp.
    x: OVector<f32, Dim>,
    /// Activation at current time stamp, inside a residual branch.
    xb: OVector<f32, Dim>,
    /// Additional buffer for convenience.
    xb2: OVector<f32, Dim>,
    /// Buffer for hidden dimension in the Feed-Forward net.
    hb: OVector<f32, HiddenDim>,
    /// Another buffer for hidden dimension in the Feed-Forward net.
    hb2: OVector<f32, HiddenDim>,
    /// Query.
    q: OVector<f32, Dim>,
    /// Scores/attention values.
    att: OMatrix<f32, SeqLen, NumHeads>,
    /// Output logits.
    logits: OVector<f32, SeqLen>,
    // KV cache. Each Vec contains `layer` elements.
    key_cache: Vec<OMatrix<f32, Dim, SeqLen>>,
    value_cache: Vec<OMatrix<f32, Dim, SeqLen>>,
}

pub struct Transformer {
    /// The hyperparameters of the architecture (the blueprint).
    config: Llama2Config,
    /// The weights of the model.
    weights: TransformerWeights,
    /// Buffer of the "wave" of activations in the forward pass.
    state: RunState,
}

impl Transformer {
    pub fn new(config: Llama2Config, weights: TransformerWeights) -> Self {
        Self {
            state: RunState::new(&config),
            config,
            weights,
        }
    }

    pub fn logits_mut(&mut self) -> &mut OVector<f32, SeqLen> {
        &mut self.state.logits
    }
}

impl RunState {
    pub fn new(config: &Llama2Config) -> Self {
        let kv_dim = (config.hidden_size * config.num_key_value_heads) / config.num_attention_heads;
        Self {
            x: DVector::zeros(config.hidden_size),
            xb: DVector::zeros(config.hidden_size),
            xb2: DVector::zeros(config.hidden_size),
            hb: DVector::zeros(config.intermediate_size),
            hb2: DVector::zeros(config.intermediate_size),
            q: DVector::zeros(config.hidden_size),
            key_cache: (0..config.num_hidden_layers)
                .map(|_| DMatrix::zeros(kv_dim, config.max_position_embeddings))
                .collect(),
            value_cache: (0..config.num_hidden_layers)
                .map(|_| DMatrix::zeros(kv_dim, config.max_position_embeddings))
                .collect(),
            att: DMatrix::zeros(config.max_position_embeddings, config.num_attention_heads),
            logits: DVector::zeros(config.vocab_size),
        }
    }
}

/*
 *
 *
 * Neural net blocks. The dynamics of the Transformer.
 *
 *
 */
/// Implementation of the Root Mean Square Normalization.
///
/// This implementation of the RMS normalization from the "Root Mean Square
/// Normalization" paper by Zhang & Sennrich.
fn rms_norm<SW: Storage<f32, Dyn>>(
    out: &mut DVector<f32>,
    a: &DVector<f32>,
    w: &Vector<f32, Dyn, SW>,
) {
    const NUDGE_FACTOR: f32 = 1.0e-5;
    let rms = 1.0 / (a.norm_squared() / (a.nrows() as f32) + NUDGE_FACTOR).sqrt();
    out.zip_zip_apply(a, w, |o, a, w| *o = (a * rms) * w);
}

/// The softmax function.
///
/// Converts a set of real number into a probability distribution.
/// See <https://fr.wikipedia.org/wiki/Fonction_softmax>
pub fn softmax<S: StorageMut<f32, Dyn>>(vals: &mut Vector<f32, Dyn, S>) {
    // Note that llama2.c also introduces a bias based on the max value
    // to improve numerical stability. So it is effectively computing:
    // softmax(z) = (e^z - max) / (e^z - max).sum()
    let max_val = vals.max();
    let mut sum = 0.0;

    vals.apply(|x| {
        *x = (*x - max_val).exp();
        sum += *x;
    });

    *vals /= sum;
}

/// Most expensive part of the inference.
fn matmul<SOut: StorageMut<f32, Dyn>>(
    out: &mut Vector<f32, Dyn, SOut>,
    x: &DVector<f32>,
    w: &DMatrix<f32>,
) {
    out.gemv(1.0, w, x, 0.0);
}

impl Transformer {
    pub fn forward(&mut self, token: usize, pos: usize) {
        // A few convenience variables.
        let config = &self.config;
        let w = &self.weights;
        let s = &mut self.state;
        let dim = config.hidden_size;
        // This is the number of key/value heads multiplied by the size of a query head: NumKvHeadsTimesHeadSize
        let kv_dim = (config.hidden_size * config.num_key_value_heads) / config.num_attention_heads;
        // The number of embedding vector elements associated to each query head.
        let head_size = dim / config.num_attention_heads;

        // Copy the token embedding into x.
        s.x.copy_from(&w.token_embd.column(token));

        // Forward all the layers.
        for l in 0..config.num_hidden_layers {
            let wl = &w.layers[l];

            // RMS norm before attention.
            // See https://youtu.be/Mn_9W1nCFLo?si=Ogz_O_6LUsumWovB&t=1367
            rms_norm(&mut s.xb, &s.x, &wl.attn_norm);

            // Key and value point to the KV cache.
            let mut k_cache = s.key_cache[l].column_mut(pos);
            let mut v_cache = s.value_cache[l].column_mut(pos);

            // qkv matmuls for this position.
            // This is self-attention, so `xb` is used for query, key, and value.
            // These are essentially one row of Q’, K’, V’ from https://youtu.be/Mn_9W1nCFLo?si=7B_g41B2iGZ5238a&t=2422
            // Note that despite keys/values having different number of heads as queries, the dimension of
            // each k/v head are the same as the query heads. The dimension change happens through the
            // multiplication by the weight matrices wk/wv.
            matmul(&mut s.q, &s.xb, &wl.attn_q);
            matmul(&mut k_cache, &s.xb, &wl.attn_k);
            matmul(&mut v_cache, &s.xb, &wl.attn_v);

            // Rotary Positional Encoding (RoPE).
            Self::rotary_positional_encoding(&mut s.q, &mut k_cache, head_size, dim, kv_dim, pos);

            // Batched multi-query attention.
            Self::attention(config, s, w, pos, l);

            // Residual connection back into x.
            // See the LLama graph on the right: https://youtu.be/Mn_9W1nCFLo?si=XMDdHlXxON2QhFCd&t=320
            // This step is the first big circled +
            s.x += &s.xb2;

            // RMSnorm before feed-forward.
            // /!\ xb changes semantic again. It now contains the normalized {attention output+input}.
            rms_norm(&mut s.xb, &s.x, &wl.ffn_norm);

            // Feed-forward.
            Self::ffn_silu(s, wl);

            // Residual connection.
            s.x += &s.xb2;
            // Loop on the next layer. This layer’s output is the next layer’s input.
        }

        // Final rmsnorm.
        // This is the top-most rmsnorm from https://youtu.be/Mn_9W1nCFLo?si=KO-aBXZo0DqCL4Qs&t=275
        // (diagram on the right).
        rms_norm(&mut s.xb, &s.x, &w.output_norm);

        // Classifier into logits.
        // This is the final "Linear" part from https://youtu.be/Mn_9W1nCFLo?si=-GT74rBY6j5TbbBO&t=275
        matmul(&mut s.logits, &s.xb, &w.output);
    }

    // Rotary Positional Encoding (RoPE): complex-valued rotate q and k in each head.
    pub fn rotary_positional_encoding(
        q: &mut DVector<f32>,
        k: &mut DVectorViewMut<f32>,
        head_size: usize,
        dim: usize,
        kv_dim: usize,
        pos: usize,
    ) {
        for i in (0..dim).step_by(2) {
            // For RoPE, we have one rotation matrix like https://youtu.be/Mn_9W1nCFLo?si=GLIXuFLGVG8q6v2u&t=1963
            // for each head. So we need to transform `i` into the corresponding index within
            // the head.
            let head_dim = (i % head_size) as f32;
            // Not that the formulae from the video linked above would be:
            //     10000.0.powf(-2.0 * ((i / 2) as f32 - 1.0) / dim as f32)
            // Although in the paper shown in the video, their index is 1-based which his why thy
            // have to subtract 1.0 whereas we don’t need to.The `i / 2` and multiplication by 2.0
            // are both accounted for by stepping only on even values for `i`.
            // Therefore, the formulae below is equivalent to the RoPE paper’s formulae.
            let theta = 10000.0_f32.powf(-head_dim / head_size as f32);
            let m_theta = pos as f32 * theta;
            let rot = Rotation2::new(m_theta);

            let qi = vector![q[i], q[i + 1]];
            let mut out_q = q.fixed_rows_mut::<2>(i);
            out_q.copy_from(&(rot * qi));

            // When i >= kv_dim, we are done rotating all the elements from the keys. That’s
            // because there are less key heads than query heads, but each key head sub-vector has
            // the same dimension as the query head (they loose dimension when multiplied with the
            // key weight matrices).
            if i < kv_dim {
                let ki = vector![k[i], k[i + 1]];
                let mut out_k = k.fixed_rows_mut::<2>(i);
                out_k.copy_from(&(rot * ki));
            }
        }
    }

    fn attention(
        config: &Llama2Config,
        s: &mut RunState,
        w: &TransformerWeights,
        pos: usize,
        l: usize,
    ) {
        // The number of embedding vector elements associated to each query head.
        let head_size = config.hidden_size / config.num_attention_heads;
        // The number of query head associated to one key/value head.
        let kv_mul = config.num_attention_heads / config.num_key_value_heads;

        // Multihead attention. Iterate over all head.
        for h in 0..config.num_attention_heads {
            // Get the query vector for this head.
            let q = s.q.rows(h * head_size, head_size);
            // Attention scores for this head.
            let mut att = s.att.column_mut(h);

            // Iterate over all timesteps (tokens in the sequence), including the current one, but
            // not past the current one due to causality.
            // See the KV cache explanation there: https://youtu.be/Mn_9W1nCFLo?si=3n4GH9f2OzMb5Np0&t=2940
            // -> This is iterating through all the green columns (from K^t) that are the rotated
            //    (by RoPE). The values set in this loop into the `att` variable here (attention
            //    scores) are the elements in the pink row (at the bottom of the QK^t matrix) divide
            //    by sqrt(head_size) (in other words, this is what’s given to softmax afterward.
            for t in 0..=pos {
                // Get the key vector for this head and at this timestep.
                let k = s.key_cache[l].column(t);
                let k_head = k.rows((h / kv_mul) * head_size, head_size);

                // Calculate the attention score as the dot product of q and k.
                let mut score = q.dot(&k_head);
                score /= (head_size as f32).sqrt();
                // Save the score to the attention buffer.
                att[t] = score;
            }

            // Softmax the scores to get attention weights from 0..=pos inclusively.
            softmax(&mut att.rows_mut(0, pos + 1));

            // Weighted sum of the values, store back into xb.
            // /!\ xb is now changing semantic, storing the weighted sums for all the heads.
            //       Now xb contains the "Attention 4" row from https://youtu.be/Mn_9W1nCFLo?si=550ar5aUg1I1k60l&t=2940.
            let mut xb = s.xb.rows_mut(h * head_size, head_size);
            xb.fill(0.0);
            for t in 0..=pos {
                let v = s.value_cache[l].column(t);
                let v_head = v.rows((h / kv_mul) * head_size, head_size);
                xb.axpy(att[t], &v_head, 1.0);
            }
        }

        // Final matmul to get the output of the attention.
        matmul(&mut s.xb2, &s.xb, &w.layers[l].attn_output);
    }

    fn ffn_silu(s: &mut RunState, wl: &TransformerLayerWeights) {
        // We have: self.w2(F.silu(self.w1(x)) * self.w3(x)) first calculate self.w1(x) and
        // self.w3(x)
        //
        // For this part, see https://youtu.be/Mn_9W1nCFLo?si=Ub9m1NeAzkmn-G8G&t=3973
        // We have: w1 := W, w3 := V, w2 := W2
        s.hb.gemv(1.0, &wl.ffn_gate, &s.xb, 0.0);
        s.hb2.gemv(1.0, &wl.ffn_up, &s.xb, 0.0);

        // SwiGLU non-linearity.
        fn swish(x: f32, beta: f32) -> f32 {
            // This is the swish function from https://youtu.be/Mn_9W1nCFLo?si=LT6puSAfzgpP6ydz&t=3973
            x / (1.0 + (-beta * x).exp())
        }

        s.hb.zip_apply(&s.hb2, |h, h2| *h = h2 * swish(*h, 1.0));

        // Final matmul to get the output of the feed-forward net.
        matmul(&mut s.xb2, &s.hb, &wl.ffn_down);
    }
}
