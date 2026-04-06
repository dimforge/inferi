/*
 * NOTE: this file is largely adapted from
 *       https://github.com/huggingface/candle/blob/d7c5c8aba502ff9a0c8ac6eff23e0cf07d6e3342/candle-examples/examples/whisper/main.rs
 */

// https://github.com/openai/whisper/blob/main/whisper/model.py/rgs
// TODO:
// - Batch size greater than 1.
// - More token filters (SuppressBlanks, ApplyTimestampRules).

use anyhow::{Error as E, Result};
use inferi::context::LlmContext;
use inferi::models::llama2::cpu::softmax;
use inferi::models::whisper::WhisperConfig;
use inferi::models::whisper::{self as m, model::Whisper};
use inferi::re_exports::tokenizers::Tokenizer;
use inferi::re_exports::vortx::tensor::Tensor;
use inferi::tensor_cache::CachedTensor;
use khal::backend::{GpuBackend, GpuBackendError};
use khal::BufferUsages;
use nalgebra::DVector;
use rand::distributions::{Distribution, WeightedIndex};
use rand::SeedableRng;

pub enum Model {
    Normal(Whisper),
    #[allow(dead_code)]
    Quantized(Whisper), // TODO: m::quantized_model::Whisper),
}

// Maybe we should use some traits rather than doing the dispatch for all these.
impl Model {
    pub fn config(&self) -> &WhisperConfig {
        match self {
            Self::Normal(m) => &m.config,
            Self::Quantized(m) => &m.config,
        }
    }

    pub async fn encoder_forward(
        &mut self,
        ctx: &mut LlmContext<'_>,
        x: &Tensor<f32>,
        flush: bool,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        match self {
            Self::Normal(m) => m.encoder.forward(ctx, x, flush).await,
            Self::Quantized(m) => m.encoder.forward(ctx, x, flush).await,
        }
    }

    pub fn decoder_forward(
        &mut self,
        ctx: &mut LlmContext,
        x: &Tensor<u32>,
        xa: &Tensor<f32>,
        flush: bool,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        match self {
            Self::Normal(m) => m.decoder.forward(ctx, x, xa, flush),
            Self::Quantized(m) => m.decoder.forward(ctx, x, xa, flush),
        }
    }

    pub fn decoder_final_linear(
        &self,
        ctx: &mut LlmContext,
        x: &Tensor<f32>,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        match self {
            Self::Normal(m) => m.decoder.final_linear(ctx, x),
            Self::Quantized(m) => m.decoder.final_linear(ctx, x),
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct DecodingResult {
    pub tokens: Vec<u32>,
    pub text: String,
    pub avg_logprob: f64,
    pub no_speech_prob: f64,
    pub temperature: f64,
    pub compression_ratio: f64,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Segment {
    pub start: f64,
    pub duration: f64,
    pub dr: DecodingResult,
}

pub struct WhisperChat {
    model: Model,
    rng: rand::rngs::StdRng,
    task: Option<Task>,
    timestamps: bool,
    #[allow(dead_code)]
    max_initial_timestamp_index: Option<u32>,
    verbose: bool,
    tokenizer: Tokenizer,
    suppress_tokens: Tensor<f32>,
    sot_token: u32,
    transcribe_token: u32,
    translate_token: u32,
    eot_token: u32,
    no_speech_token: u32,
    no_timestamps_token: u32,
    language_token: Option<u32>,
}

impl WhisperChat {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        backend: &GpuBackend,
        language_token: Option<u32>,
        task: Option<Task>,
        timestamps: bool,
        max_initial_timestamp_index: Option<u32>,
        verbose: bool,
    ) -> Result<Self> {
        let no_timestamps_token = token_id(&tokenizer, m::NO_TIMESTAMPS_TOKEN)?;
        // Suppress the notimestamps token when in timestamps mode.
        // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L452
        let suppress_tokens: Vec<f32> = (0..model.config().vocab_size as u32)
            .map(|i| {
                if model.config().suppress_tokens.contains(&i)
                    || timestamps && i == no_timestamps_token
                {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        let suppress_tokens =
            Tensor::vector(backend, suppress_tokens.as_slice(), BufferUsages::STORAGE)?;
        let sot_token = token_id(&tokenizer, m::SOT_TOKEN)?;
        let transcribe_token = token_id(&tokenizer, m::TRANSCRIBE_TOKEN)?;
        let translate_token = token_id(&tokenizer, m::TRANSLATE_TOKEN)?;
        let eot_token = token_id(&tokenizer, m::EOT_TOKEN)?;
        let no_speech_token = m::NO_SPEECH_TOKENS
            .iter()
            .find_map(|token| token_id(&tokenizer, token).ok());
        let no_speech_token = match no_speech_token {
            None => anyhow::bail!("unable to find any non-speech token"),
            Some(n) => n,
        };
        Ok(Self {
            model,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            tokenizer,
            task,
            timestamps,
            max_initial_timestamp_index,
            verbose,
            suppress_tokens,
            sot_token,
            transcribe_token,
            translate_token,
            eot_token,
            no_speech_token,
            language_token,
            no_timestamps_token,
        })
    }

    async fn decode(
        &mut self,
        ctx: &mut LlmContext<'_>,
        mel: &Tensor<f32>,
        t: f64,
    ) -> Result<DecodingResult> {
        let audio_features = self.model.encoder_forward(ctx, mel, true).await?;

        if self.verbose {
            println!("audio features: {:?}", audio_features.layout());
        }
        let sample_len = self.model.config().max_target_positions / 2;
        let mut sum_logprob = 0f64;
        let mut no_speech_prob = f64::NAN;
        let mut tokens = vec![self.sot_token];
        if let Some(language_token) = self.language_token {
            tokens.push(language_token);
        }
        match self.task {
            None | Some(Task::Transcribe) => tokens.push(self.transcribe_token),
            Some(Task::Translate) => tokens.push(self.translate_token),
        }
        if !self.timestamps {
            tokens.push(self.no_timestamps_token);
        }

        for i in 0..sample_len {
            let tokens_t = Tensor::vector(ctx.backend, tokens.as_slice(), BufferUsages::STORAGE)?;

            // The model expects a batch dim but this inference loop does not handle
            // it so we add it at this point.
            let tokens_t = tokens_t.as_view().unsqueeze(0);
            let tokens_t = ctx.contiguous(tokens_t)?; // TODO PERF: avoid contiguous
            let ys = self
                .model
                .decoder_forward(ctx, &tokens_t, &audio_features, i == 0)?;
            // println!("Decoder output: {:?}", ys.as_view().layout());

            // {
            //     let mut data = vec![0.0; ys.len() as usize];
            //     ctx.backend
            //         .slow_read_buffer(ys.buffer(), &mut data)
            //         .await
            //         .unwrap();
            //     println!("data shape: {:?}", ys.as_view().layout());
            //     for i in 0..77 {
            //         // 60..70 {
            //         println!(
            //             "######### data [{i}]: {:?}",
            //             &data[10 * i..(10 * (i + 1)).min(data.len())]
            //         );
            //     }
            //     std::process::abort();
            // }

            // Extract the no speech probability on the first iteration by looking at the first
            // token logits and the probability for the according token.
            if i == 0 {
                let ys_row0 = ctx.contiguous(ys.narrow(0, 0, 1))?; // TODO PERF: avoid contiguous
                let logits = self.model.decoder_final_linear(ctx, &ys_row0)?;
                let mut logits = ctx.contiguous(logits.as_view().index(0).index(0))?; // TODO PERF: avoid contiguous

                ctx.softmax_rows(&mut logits)?;

                // TODO PERF: avoid contiguous
                let probs = logits;

                // TODO PERF: only read back the single probability we are interested in, instead of the whole buffer.
                let probs = ctx.slow_read_vec(probs.buffer()).await?;
                no_speech_prob = probs[self.no_speech_token as usize] as f64;
            }

            let [_, seq_len, _, _] = ys.layout().size;
            let ys_slice = ys.narrow(0, 0, 1).narrow(1, seq_len - 1, 1);
            let ys_cont = ctx.contiguous(ys_slice)?; // TODO PERF: avoid contiguous
            let logits = self.model.decoder_final_linear(ctx, &ys_cont)?;

            // TODO: not sure about the indexing here.
            // TODO PERF: avoid contiguous
            let logits = ctx.contiguous(logits.as_view().index(0).index(0))?;

            // println!("|!|!|!|!|!|!|!|!|!|!|! IGNORING TIMESTAMP RULES FOR NOW. NEEDS FIXING IN SLAI |!|!|!|!|!|!|!|!|!");
            // // Apply timestamp rules when timestamps are enabled
            // let logits = if self.timestamps {
            //     self.apply_timestamp_rules(ctx, &logits, &tokens).await?
            // } else {
            //     logits
            // };

            let mut logits = ctx.add(&logits, &self.suppress_tokens)?;

            let next_token = if t > 0f64 {
                let mut probs = DVector::from(ctx.slow_read_vec(logits.buffer()).await?);
                probs /= t as f32;
                softmax(&mut probs); // TODO PERF: run on GPU?
                let distr = WeightedIndex::new(&probs)?;
                distr.sample(&mut self.rng) as u32
            } else {
                let logits_v: Vec<f32> = ctx.slow_read_vec(logits.buffer()).await?;
                logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, u), (_, v)| u.total_cmp(v))
                    .map(|(i, _)| i as u32)
                    .unwrap()
            };

            tokens.push(next_token);

            // println!("Softmax shape: {:?}", logits.layout());
            ctx.softmax_rows(&mut logits)?;

            let probs = logits;
            let prob = ctx.slow_read_vec(probs.buffer()).await?;
            let prob = prob[next_token as usize] as f64;
            if next_token == self.eot_token
                || tokens.len() > self.model.config().max_target_positions
            {
                break;
            }
            sum_logprob += prob.ln();
        }

        let text = self.tokenizer.decode(&tokens, true).map_err(E::msg)?;
        let avg_logprob = sum_logprob / tokens.len() as f64;

        Ok(DecodingResult {
            tokens,
            text,
            avg_logprob,
            no_speech_prob,
            temperature: t,
            compression_ratio: f64::NAN,
        })
    }

    async fn decode_with_fallback(
        &mut self,
        ctx: &mut LlmContext<'_>,
        segment: &Tensor<f32>,
    ) -> Result<DecodingResult> {
        for (i, &t) in m::TEMPERATURES.iter().enumerate() {
            let dr: Result<DecodingResult> = self.decode(ctx, segment, t).await;
            if i == m::TEMPERATURES.len() - 1 {
                return dr;
            }
            // On errors, we try again with a different temperature.
            match dr {
                Ok(dr) => {
                    let needs_fallback = dr.compression_ratio > m::COMPRESSION_RATIO_THRESHOLD
                        || dr.avg_logprob < m::LOGPROB_THRESHOLD;
                    if !needs_fallback || dr.no_speech_prob > m::NO_SPEECH_THRESHOLD {
                        return Ok(dr);
                    }
                }
                Err(err) => {
                    println!("Error running at {t}: {err}")
                }
            }
        }
        unreachable!()
    }

    #[allow(dead_code)]
    async fn apply_timestamp_rules(
        &self,
        ctx: &mut LlmContext<'_>,
        input_logits: &Tensor<f32>,
        tokens: &[u32],
    ) -> Result<CachedTensor<f32>> {
        let timestamp_begin = self.no_timestamps_token + 1;
        let vocab_size = self.model.config().vocab_size as u32;

        // ========== SETUP: Extract sampled tokens for analysis ==========
        let sample_begin = if self.language_token.is_some() { 3 } else { 2 };
        let sampled_tokens = if tokens.len() > sample_begin {
            &tokens[sample_begin..]
        } else {
            &[]
        };

        let mut masks = Vec::new();
        // Pre-allocate reusable mask buffer to avoid repeated allocations
        let mut mask_buffer = vec![0.0f32; vocab_size as usize];

        // ========== RULE 1: Timestamp pairing constraints ==========
        // Timestamps must come in pairs, except directly before EOT
        if !sampled_tokens.is_empty() {
            let last_was_timestamp = sampled_tokens
                .last()
                .map(|&t| t >= timestamp_begin)
                .unwrap_or(false);

            let penultimate_was_timestamp = if sampled_tokens.len() >= 2 {
                sampled_tokens[sampled_tokens.len() - 2] >= timestamp_begin
            } else {
                false
            };

            if last_was_timestamp {
                if penultimate_was_timestamp {
                    // Has to be non-timestamp - suppress timestamp tokens
                    for i in 0..vocab_size {
                        mask_buffer[i as usize] = if i >= timestamp_begin {
                            f32::NEG_INFINITY
                        } else {
                            0.0
                        };
                    }
                    masks.push(Tensor::vector(
                        ctx.backend,
                        mask_buffer.as_slice(),
                        BufferUsages::STORAGE,
                    )?);
                } else {
                    // Cannot be normal text tokens - suppress everything before EOT
                    for i in 0..vocab_size {
                        mask_buffer[i as usize] = if i < self.eot_token {
                            f32::NEG_INFINITY
                        } else {
                            0.0
                        };
                    }
                    masks.push(Tensor::vector(
                        ctx.backend,
                        mask_buffer.as_slice(),
                        BufferUsages::STORAGE,
                    )?);
                }
            }

            // ========== RULE 2: Non-decreasing timestamp constraint ==========
            // Timestamps shouldn't decrease; forbid timestamp tokens smaller than the last
            let timestamp_tokens: Vec<u32> = sampled_tokens
                .iter()
                .filter(|&&t| t >= timestamp_begin)
                .cloned()
                .collect();

            if !timestamp_tokens.is_empty() {
                let timestamp_last = if last_was_timestamp && !penultimate_was_timestamp {
                    *timestamp_tokens.last().unwrap()
                } else {
                    timestamp_tokens.last().unwrap() + 1
                };

                for i in 0..vocab_size {
                    mask_buffer[i as usize] = if i >= timestamp_begin && i < timestamp_last {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    };
                }
                masks.push(Tensor::vector(
                    ctx.backend,
                    mask_buffer.as_slice(),
                    BufferUsages::STORAGE,
                )?);
            }
        }

        // ========== RULE 3: Force initial timestamp ==========
        // At the beginning, suppress generating non-timestamp tokens
        if tokens.len() == sample_begin {
            for i in 0..vocab_size {
                mask_buffer[i as usize] = if i < timestamp_begin {
                    f32::NEG_INFINITY
                } else {
                    0.0
                };
            }
            masks.push(Tensor::vector(
                ctx.backend,
                mask_buffer.as_slice(),
                BufferUsages::STORAGE,
            )?);

            // Apply the max_initial_timestamp constraint
            if let Some(max_initial_timestamp_index) = self.max_initial_timestamp_index {
                let last_allowed = timestamp_begin + max_initial_timestamp_index;
                if last_allowed < vocab_size {
                    for i in 0..vocab_size {
                        mask_buffer[i as usize] = if i > last_allowed {
                            f32::NEG_INFINITY
                        } else {
                            0.0
                        };
                    }
                    masks.push(Tensor::vector(
                        ctx.backend,
                        mask_buffer.as_slice(),
                        BufferUsages::STORAGE,
                    )?);
                }
            }
        }

        // ========== APPLY MASKS: Apply all constraint masks ==========
        let mut logits = ctx.clone(input_logits)?;
        for mask in masks {
            logits = ctx.add(&logits, &mask)?;
        }

        // ========== RULE 4: Probability-based timestamp preference ==========
        // If sum of probability over timestamps is above any other token, sample timestamp
        let mut log_probs = ctx.clone(&logits)?; // TODO PERF: avoid the clone?
        ctx.log_softmax_rows(&mut log_probs)?;

        // Extract timestamp and text log probabilities
        let timestamp_log_probs =
            log_probs.narrow(0, timestamp_begin, vocab_size - timestamp_begin);

        let text_log_probs = log_probs.narrow(0, 0, timestamp_begin);

        // FIXME: for simplicity, we are running the following on the CPU.
        //        We should implement the logsumexp operation on the GPU.
        let timestamp_log_probs = ctx.contiguous(timestamp_log_probs)?;
        let text_log_probs = ctx.contiguous(text_log_probs)?;
        let timestamp_log_probs =
            DVector::from(ctx.slow_read_vec(timestamp_log_probs.buffer()).await?);
        let text_log_probs = DVector::from(ctx.slow_read_vec(text_log_probs.buffer()).await?);

        // Implement logsumexp for timestamp tokens (numerically stable)
        let timestamp_logprob = {
            let max_val = timestamp_log_probs.max();
            let shifted = timestamp_log_probs.map(|e| e - max_val);
            let exp_shifted = shifted.map(|e| e.exp());
            let sum_exp = exp_shifted.sum();
            let log_sum = sum_exp.ln();
            max_val + log_sum
        };

        // Get max text token log probability
        let max_text_token_logprob: f32 = text_log_probs.max();

        // Compare in log space
        if timestamp_logprob > max_text_token_logprob {
            // Only consider timestamp tokens
            for i in 0..vocab_size {
                mask_buffer[i as usize] = if i < timestamp_begin {
                    f32::NEG_INFINITY
                } else {
                    0.0
                };
            }
            let mask_tensor =
                ctx.tensor([1, mask_buffer.len() as u32, 1, 1], mask_buffer.as_slice())?;
            logits = ctx.add(&logits, &mask_tensor)?;
        }

        Ok(logits)
    }

    pub async fn run(
        &mut self,
        ctx: &mut LlmContext<'_>,
        mel: &Tensor<f32>,
    ) -> Result<Vec<Segment>> {
        let [_, _, content_frames, _] = mel.layout().size;
        let mut seek = 0;
        let mut segments = vec![];
        while seek < content_frames {
            let start = web_time::Instant::now();
            let time_offset = (seek as usize * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
            let segment_size = usize::min((content_frames - seek) as usize, m::N_FRAMES);
            let mel_segment = mel.narrow(2, seek, segment_size as u32);
            let segment_duration = (segment_size * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;

            let mel_segment = ctx.contiguous(mel_segment)?; // TODO PERF: avoid the contiguous?
            let dr = self.decode_with_fallback(ctx, &mel_segment).await?;
            seek += segment_size as u32;
            if dr.no_speech_prob > m::NO_SPEECH_THRESHOLD && dr.avg_logprob < m::LOGPROB_THRESHOLD {
                println!("no speech detected, skipping {seek} {dr:?}");
                continue;
            }
            let segment = Segment {
                start: time_offset,
                duration: segment_duration,
                dr,
            };
            if self.timestamps {
                println!(
                    "{:.1}s -- {:.1}s",
                    segment.start,
                    segment.start + segment.duration,
                );
                let mut tokens_to_decode = vec![];
                let mut prev_timestamp_s = 0f32;
                for &token in segment.dr.tokens.iter() {
                    if token == self.sot_token || token == self.eot_token {
                        continue;
                    }
                    // The no_timestamp_token is the last before the timestamp ones.
                    if token > self.no_timestamps_token {
                        let timestamp_s = (token - self.no_timestamps_token + 1) as f32 / 50.;
                        if !tokens_to_decode.is_empty() {
                            let text = self
                                .tokenizer
                                .decode(&tokens_to_decode, true)
                                .map_err(E::msg)?;
                            println!("  {:.1}s-{:.1}s: {}", prev_timestamp_s, timestamp_s, text);
                            tokens_to_decode.clear()
                        }
                        prev_timestamp_s = timestamp_s;
                    } else {
                        tokens_to_decode.push(token)
                    }
                }
                if !tokens_to_decode.is_empty() {
                    let text = self
                        .tokenizer
                        .decode(&tokens_to_decode, true)
                        .map_err(E::msg)?;
                    if !text.is_empty() {
                        println!("  {:.1}s-...: {}", prev_timestamp_s, text);
                    }
                    tokens_to_decode.clear()
                }
            } else {
                println!(
                    "{:.1}s -- {:.1}s: {}",
                    segment.start,
                    segment.start + segment.duration,
                    segment.dr.text,
                )
            }
            if self.verbose {
                println!("{seek}: {segment:?}, in {:?}", start.elapsed());
            }
            segments.push(segment)
        }
        Ok(segments)
    }
}

pub fn token_id(tokenizer: &Tokenizer, token: &str) -> anyhow::Result<u32> {
    match tokenizer.token_to_id(token) {
        None => anyhow::bail!("no token-id for {token}"),
        Some(id) => Ok(id),
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub enum Task {
    Transcribe,
    Translate,
}
