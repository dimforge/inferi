// Ported from C++ https://github.com/YavorGIvanov/sam.cpp/tree/master
// (MIT license)

mod decode_mask;
mod encode_image;
mod encode_prompt;
mod fill_dense_pe;
mod io;
mod layernorm2d;
mod post_process_mask;

pub use decode_mask::sam_decode_mask;
pub use encode_image::sam_encode_image;
pub use encode_prompt::{sam_encode_prompt, SamPromptEncoderResult};
pub use fill_dense_pe::sam_fill_dense_pe;
pub use io::SamGgmlFile;
pub use post_process_mask::sam_postprocess_masks;

use crate::gguf::Gguf;
use crate::safetensor::SafeTensorExt;
use khal::backend::GpuBackend;
use khal::BufferUsages;
use nalgebra::DMatrix;
use safetensors::SafeTensors;
use vortx::tensor::{Tensor, TensorBuilder};

pub struct SamImage {
    pub pixels: DMatrix<[f32; 3]>,
}

pub struct SamImageU8 {
    pub nx: usize,
    pub ny: usize,
    pub data: Vec<u8>,
}

// Defaults to hparams for ViT-B SAM
#[derive(Debug)]
pub struct SamHParams {
    pub n_enc_state: u32,
    pub n_enc_layer: u32,
    pub n_enc_head: u32,
    pub n_enc_out_chans: u32,
    pub n_pt_embd: u32,
    pub n_dec_heads: u32,
    pub ftype: u32,
    pub mask_threshold: f32,
    pub iou_threshold: f32,
    pub stability_score_threshold: f32,
    pub stability_score_offset: f32,
    pub eps: f32,
    pub eps_decoder_transformer: f32,
}

impl Default for SamHParams {
    fn default() -> Self {
        Self {
            n_enc_state: 768,
            n_enc_layer: 12,
            n_enc_head: 12,
            n_enc_out_chans: 256,
            n_pt_embd: 4,
            n_dec_heads: 8,
            ftype: 1,
            mask_threshold: 0.0,
            iou_threshold: 0.85,
            stability_score_threshold: 0.90,
            stability_score_offset: 1.0,
            eps: 1.0e-6,
            eps_decoder_transformer: 1.0e-5,
        }
    }
}

impl SamHParams {
    pub fn n_enc_head_dim(&self) -> u32 {
        self.n_enc_state / self.n_enc_head
    }

    pub fn n_img_size(&self) -> u32 {
        1024
    }

    pub fn n_window_size(&self) -> u32 {
        14
    }

    pub fn n_patch_size(&self) -> u32 {
        16
    }

    pub fn n_img_embd(&self) -> u32 {
        self.n_img_size() / self.n_patch_size()
    }

    pub fn global_attn_indices(&self) -> [u32; 4] {
        match self.n_enc_state {
            768 => [2, 5, 8, 11],
            1024 => [5, 11, 17, 23],
            1280 => [7, 15, 23, 31],
            _ => panic!("Unsupported n_enc_state: {}", self.n_enc_state),
        }
    }

    pub fn is_global_attn(&self, layer: u32) -> bool {
        self.global_attn_indices().contains(&layer)
    }
}

pub struct SamLayerEnc {
    pub norm1_w: Tensor<f32>,
    pub norm1_b: Tensor<f32>,

    pub rel_pos_w: Tensor<f32>,
    pub rel_pos_h: Tensor<f32>,

    pub qkv_w: Tensor<f32>,
    pub qkv_b: Tensor<f32>,

    pub proj_w: Tensor<f32>,
    pub proj_b: Tensor<f32>,

    pub norm2_w: Tensor<f32>,
    pub norm2_b: Tensor<f32>,

    pub mlp_lin1_w: Tensor<f32>,
    pub mlp_lin1_b: Tensor<f32>,

    pub mlp_lin2_w: Tensor<f32>,
    pub mlp_lin2_b: Tensor<f32>,
}

pub struct SamEncoderImage {
    pub pe: Tensor<f32>,
    pub proj_w: Tensor<f32>,
    pub proj_b: Tensor<f32>,
    pub neck_conv_0: Tensor<f32>,
    pub neck_norm_0_w: Tensor<f32>,
    pub neck_norm_0_b: Tensor<f32>,
    pub neck_conv_1: Tensor<f32>,
    pub neck_norm_1_w: Tensor<f32>,
    pub neck_norm_1_b: Tensor<f32>,
    pub layers: Vec<SamLayerEnc>,
}

pub struct SamEncoderPrompt {
    pub pe: Tensor<f32>,
    pub not_a_pt_embd_w: Tensor<f32>,
    pub pt_embd: Vec<Tensor<f32>>,
    pub no_mask_embd_w: Tensor<f32>,
}

pub struct SamLayerDecTransformerAttn {
    // q_proj
    pub q_w: Tensor<f32>,
    pub q_b: Tensor<f32>,

    // k_proj
    pub k_w: Tensor<f32>,
    pub k_b: Tensor<f32>,

    // v_proj
    pub v_w: Tensor<f32>,
    pub v_b: Tensor<f32>,

    // out_proj
    pub out_w: Tensor<f32>,
    pub out_b: Tensor<f32>,
}

pub struct SamLayerDecTransformer {
    self_attn: SamLayerDecTransformerAttn,

    // norm1
    norm1_w: Tensor<f32>,
    norm1_b: Tensor<f32>,

    cross_attn_token_to_img: SamLayerDecTransformerAttn,

    // norm2
    norm2_w: Tensor<f32>,
    norm2_b: Tensor<f32>,

    // mlp.lin1
    mlp_lin1_w: Tensor<f32>,
    mlp_lin1_b: Tensor<f32>,

    // mlp.lin2
    mlp_lin2_w: Tensor<f32>,
    mlp_lin2_b: Tensor<f32>,

    // norm3
    norm3_w: Tensor<f32>,
    norm3_b: Tensor<f32>,

    // norm4
    norm4_w: Tensor<f32>,
    norm4_b: Tensor<f32>,

    cross_attn_img_to_token: SamLayerDecTransformerAttn,
}

struct SamLayerDecOutputHypernetMlps {
    // mlps_*.layers.0
    w_0: Tensor<f32>,
    b_0: Tensor<f32>,

    // mlps_*.layers.1
    w_1: Tensor<f32>,
    b_1: Tensor<f32>,

    // mlps_*.layers.2
    w_2: Tensor<f32>,
    b_2: Tensor<f32>,
}

struct SamDecoderMask {
    transformer_layers: Vec<SamLayerDecTransformer>,

    // trasnformer.final_attn_token_to_image
    transformer_final_attn_token_to_img: SamLayerDecTransformerAttn,

    // transformer.norm_final
    transformer_norm_final_w: Tensor<f32>,
    transformer_norm_final_b: Tensor<f32>,

    // output_upscaling.0
    output_upscaling_0_w: Tensor<f32>,
    output_upscaling_0_b: Tensor<f32>,

    // output_upscaling.1
    output_upscaling_1_w: Tensor<f32>,
    output_upscaling_1_b: Tensor<f32>,

    // output_upscaling.3
    output_upscaling_3_w: Tensor<f32>,
    output_upscaling_3_b: Tensor<f32>,

    // output_hypernetworks_mlps
    output_hypernet_mlps: Vec<SamLayerDecOutputHypernetMlps>,

    // iou_prediction_head.0
    iou_prediction_head_0_w: Tensor<f32>,
    iou_prediction_head_0_b: Tensor<f32>,

    // iou_prediction_head.1
    iou_prediction_head_1_w: Tensor<f32>,
    iou_prediction_head_1_b: Tensor<f32>,

    // iou_prediction_head.2
    iou_prediction_head_2_w: Tensor<f32>,
    iou_prediction_head_2_b: Tensor<f32>,

    // iou_token.weight
    iou_token_w: Tensor<f32>,

    // mask_tokens.weight
    mask_tokens_w: Tensor<f32>,
}

pub struct SamModel {
    pub hparams: SamHParams,
    enc_img: SamEncoderImage,
    enc_prompt: SamEncoderPrompt,
    dec: SamDecoderMask,
    // TODO: backend: ggml_backend_t
    // TODO: buffer: ggml_backend_buffer_t
    // TODO: ctx: ggml_contetx
    // TODO: tensors: HashMap<String, ggml_tensor*>
}

#[derive(Copy, Clone, serde::Serialize, serde::Deserialize)]
struct SamStVisionConfig {
    hidden_size: u32,         // n_enc_state
    num_hidden_layers: u32,   // n_enc_layer
    num_attention_heads: u32, // n_enc_head
    output_channels: u32,     // n_enc_out_chans
}

#[derive(Copy, Clone, serde::Serialize, serde::Deserialize)]
struct SamStPromptEncoderConfig {
    num_point_embeddings: u32, // n_pt_embd
}

#[derive(Copy, Clone, serde::Serialize, serde::Deserialize)]
struct SamStMaskDecoderConfig {
    num_attention_heads: u32, // n_dec_heads
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct SamStConfig {
    mask_decoder_config: SamStMaskDecoderConfig,
    vision_config: SamStVisionConfig,
    prompt_encoder_config: SamStPromptEncoderConfig,
    torch_dtype: String, // ftype
}

impl SamModel {
    pub fn from_safetensors(
        backend: &GpuBackend,
        st: &SafeTensors,
        config_json: &str,
    ) -> anyhow::Result<Self> {
        let storage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        let load_tensor = |name: &str| {
            let t = st
                .tensor(name)
                .unwrap()
                .to_gpu_tensor_f32_with_usage(backend, storage, false)?;
            println!("Loaded ST tensor: {} {:?}", name, t.layout());
            Ok::<_, anyhow::Error>(t)
        };

        /*
         * hparams
         */
        let hparams = {
            let config: SamStConfig = serde_json::from_str(config_json)?;

            let n_enc_state = config.vision_config.hidden_size;
            let n_enc_layer = config.vision_config.num_hidden_layers;
            let n_enc_head = config.vision_config.num_attention_heads;
            let n_enc_out_chans = config.vision_config.output_channels;
            let n_pt_embd = config.prompt_encoder_config.num_point_embeddings;
            let n_dec_heads = config.mask_decoder_config.num_attention_heads;
            let ftype = if config.torch_dtype == "float32" {
                0
            } else {
                1
            };

            SamHParams {
                n_enc_state,
                n_enc_layer,
                n_enc_head,
                n_enc_out_chans,
                n_pt_embd,
                n_dec_heads,
                ftype,
                ..Default::default()
            }
        };
        println!("Hparams: {:?}", hparams);

        /*
         * Image encoder.
         */
        let enc_img = {
            let pe = load_tensor("vision_encoder.pos_embed")?;
            let proj_w = load_tensor("vision_encoder.patch_embed.projection.weight")?;
            let proj_b = load_tensor("vision_encoder.patch_embed.projection.bias")?;
            let neck_conv_0 = load_tensor("vision_encoder.neck.conv1.weight")?;
            let neck_conv_1 = load_tensor("vision_encoder.neck.conv2.weight")?;
            let neck_norm_0_w = load_tensor("vision_encoder.neck.layer_norm1.weight")?;
            let neck_norm_0_b = load_tensor("vision_encoder.neck.layer_norm1.bias")?;
            let neck_norm_1_w = load_tensor("vision_encoder.neck.layer_norm2.weight")?;
            let neck_norm_1_b = load_tensor("vision_encoder.neck.layer_norm2.bias")?;

            let mut layers = vec![];
            for i in 0..hparams.n_enc_layer {
                let norm1_w = format!("vision_encoder.layers.{i}.layer_norm1.weight");
                let norm1_b = format!("vision_encoder.layers.{i}.layer_norm1.bias");
                let rel_pos_w = format!("vision_encoder.layers.{i}.attn.rel_pos_w");
                let rel_pos_h = format!("vision_encoder.layers.{i}.attn.rel_pos_h");
                let qkv_w = format!("vision_encoder.layers.{i}.attn.qkv.weight");
                let qkv_b = format!("vision_encoder.layers.{i}.attn.qkv.bias");
                let proj_w = format!("vision_encoder.layers.{i}.attn.proj.weight");
                let proj_b = format!("vision_encoder.layers.{i}.attn.proj.bias");
                let norm2_w = format!("vision_encoder.layers.{i}.layer_norm2.weight");
                let norm2_b = format!("vision_encoder.layers.{i}.layer_norm2.bias");
                let mlp_lin1_w = format!("vision_encoder.layers.{i}.mlp.lin1.weight");
                let mlp_lin1_b = format!("vision_encoder.layers.{i}.mlp.lin1.bias");
                let mlp_lin2_w = format!("vision_encoder.layers.{i}.mlp.lin2.weight");
                let mlp_lin2_b = format!("vision_encoder.layers.{i}.mlp.lin2.bias");
                let layer = SamLayerEnc {
                    norm1_w: load_tensor(&norm1_w)?,
                    norm1_b: load_tensor(&norm1_b)?,
                    rel_pos_w: load_tensor(&rel_pos_w)?,
                    rel_pos_h: load_tensor(&rel_pos_h)?,
                    qkv_w: load_tensor(&qkv_w)?,
                    qkv_b: load_tensor(&qkv_b)?,
                    proj_w: load_tensor(&proj_w)?,
                    proj_b: load_tensor(&proj_b)?,
                    norm2_w: load_tensor(&norm2_w)?,
                    norm2_b: load_tensor(&norm2_b)?,
                    mlp_lin1_w: load_tensor(&mlp_lin1_w)?,
                    mlp_lin1_b: load_tensor(&mlp_lin1_b)?,
                    mlp_lin2_w: load_tensor(&mlp_lin2_w)?,
                    mlp_lin2_b: load_tensor(&mlp_lin2_b)?,
                };
                layers.push(layer);
            }

            SamEncoderImage {
                pe,
                proj_w,
                proj_b,
                neck_conv_0,
                neck_norm_0_w,
                neck_norm_0_b,
                neck_conv_1,
                neck_norm_1_w,
                neck_norm_1_b,
                layers,
            }
        };

        /*
         * Prompt encoder.
         */
        let enc_prompt = {
            let pe = load_tensor("shared_image_embedding.positional_embedding")?; // TODO: is this correct?
            let not_a_pt_embd_w = load_tensor("prompt_encoder.not_a_point_embed.weight")?;
            let no_mask_embd_w = load_tensor("prompt_encoder.no_mask_embed.weight")?;

            let mut pt_embd = vec![];
            for i in 0..hparams.n_pt_embd {
                let weight = format!("prompt_encoder.point_embed.{i}.weight");
                pt_embd.push(load_tensor(&weight)?);
            }

            SamEncoderPrompt {
                pe,
                not_a_pt_embd_w,
                pt_embd,
                no_mask_embd_w,
            }
        };

        /*
         * Mask decoder.
         */
        let dec = {
            let mut transformer_layers = vec![];
            let tfm_layers_count = 2;
            for i in 0..tfm_layers_count {
                let prefix = format!("mask_decoder.transformer.layers.{i}");
                let self_attn_q_w = format!("{prefix}.self_attn.q_proj.weight");
                let self_attn_q_b = format!("{prefix}.self_attn.q_proj.bias");
                let self_attn_k_w = format!("{prefix}.self_attn.k_proj.weight");
                let self_attn_k_b = format!("{prefix}.self_attn.k_proj.bias");
                let self_attn_v_w = format!("{prefix}.self_attn.v_proj.weight");
                let self_attn_v_b = format!("{prefix}.self_attn.v_proj.bias");
                let self_attn_out_w = format!("{prefix}.self_attn.out_proj.weight");
                let self_attn_out_b = format!("{prefix}.self_attn.out_proj.bias");

                let norm1_w = format!("{prefix}.layer_norm1.weight");
                let norm1_b = format!("{prefix}.layer_norm1.bias");

                let cross_attn_token_to_img_q_w =
                    format!("{prefix}.cross_attn_token_to_image.q_proj.weight");
                let cross_attn_token_to_img_q_b =
                    format!("{prefix}.cross_attn_token_to_image.q_proj.bias");
                let cross_attn_token_to_img_k_w =
                    format!("{prefix}.cross_attn_token_to_image.k_proj.weight");
                let cross_attn_token_to_img_k_b =
                    format!("{prefix}.cross_attn_token_to_image.k_proj.bias");
                let cross_attn_token_to_img_v_w =
                    format!("{prefix}.cross_attn_token_to_image.v_proj.weight");
                let cross_attn_token_to_img_v_b =
                    format!("{prefix}.cross_attn_token_to_image.v_proj.bias");
                let cross_attn_token_to_img_out_w =
                    format!("{prefix}.cross_attn_token_to_image.out_proj.weight");
                let cross_attn_token_to_img_out_b =
                    format!("{prefix}.cross_attn_token_to_image.out_proj.bias");

                let norm2_w = format!("{prefix}.layer_norm2.weight");
                let norm2_b = format!("{prefix}.layer_norm2.bias");

                let mlp_lin1_w = format!("{prefix}.mlp.lin1.weight");
                let mlp_lin1_b = format!("{prefix}.mlp.lin1.bias");
                let mlp_lin2_w = format!("{prefix}.mlp.lin2.weight");
                let mlp_lin2_b = format!("{prefix}.mlp.lin2.bias");

                let norm3_w = format!("{prefix}.layer_norm3.weight");
                let norm3_b = format!("{prefix}.layer_norm3.bias");
                let norm4_w = format!("{prefix}.layer_norm4.weight");
                let norm4_b = format!("{prefix}.layer_norm4.bias");

                let cross_attn_img_to_token_q_w =
                    format!("{prefix}.cross_attn_image_to_token.q_proj.weight");
                let cross_attn_img_to_token_q_b =
                    format!("{prefix}.cross_attn_image_to_token.q_proj.bias");
                let cross_attn_img_to_token_k_w =
                    format!("{prefix}.cross_attn_image_to_token.k_proj.weight");
                let cross_attn_img_to_token_k_b =
                    format!("{prefix}.cross_attn_image_to_token.k_proj.bias");
                let cross_attn_img_to_token_v_w =
                    format!("{prefix}.cross_attn_image_to_token.v_proj.weight");
                let cross_attn_img_to_token_v_b =
                    format!("{prefix}.cross_attn_image_to_token.v_proj.bias");
                let cross_attn_img_to_token_out_w =
                    format!("{prefix}.cross_attn_image_to_token.out_proj.weight");
                let cross_attn_img_to_token_out_b =
                    format!("{prefix}.cross_attn_image_to_token.out_proj.bias");

                let self_attn = SamLayerDecTransformerAttn {
                    q_w: load_tensor(&self_attn_q_w)?,
                    q_b: load_tensor(&self_attn_q_b)?,
                    k_w: load_tensor(&self_attn_k_w)?,
                    k_b: load_tensor(&self_attn_k_b)?,
                    v_w: load_tensor(&self_attn_v_w)?,
                    v_b: load_tensor(&self_attn_v_b)?,
                    out_w: load_tensor(&self_attn_out_w)?,
                    out_b: load_tensor(&self_attn_out_b)?,
                };
                let cross_attn_token_to_img = SamLayerDecTransformerAttn {
                    q_w: load_tensor(&cross_attn_token_to_img_q_w)?,
                    q_b: load_tensor(&cross_attn_token_to_img_q_b)?,
                    k_w: load_tensor(&cross_attn_token_to_img_k_w)?,
                    k_b: load_tensor(&cross_attn_token_to_img_k_b)?,
                    v_w: load_tensor(&cross_attn_token_to_img_v_w)?,
                    v_b: load_tensor(&cross_attn_token_to_img_v_b)?,
                    out_w: load_tensor(&cross_attn_token_to_img_out_w)?,
                    out_b: load_tensor(&cross_attn_token_to_img_out_b)?,
                };
                let cross_attn_img_to_token = SamLayerDecTransformerAttn {
                    q_w: load_tensor(&cross_attn_img_to_token_q_w)?,
                    q_b: load_tensor(&cross_attn_img_to_token_q_b)?,
                    k_w: load_tensor(&cross_attn_img_to_token_k_w)?,
                    k_b: load_tensor(&cross_attn_img_to_token_k_b)?,
                    v_w: load_tensor(&cross_attn_img_to_token_v_w)?,
                    v_b: load_tensor(&cross_attn_img_to_token_v_b)?,
                    out_w: load_tensor(&cross_attn_img_to_token_out_w)?,
                    out_b: load_tensor(&cross_attn_img_to_token_out_b)?,
                };
                let layer = SamLayerDecTransformer {
                    self_attn,
                    norm1_w: load_tensor(&norm1_w)?,
                    norm1_b: load_tensor(&norm1_b)?,
                    cross_attn_token_to_img,
                    norm2_w: load_tensor(&norm2_w)?,
                    norm2_b: load_tensor(&norm2_b)?,
                    mlp_lin1_w: load_tensor(&mlp_lin1_w)?,
                    mlp_lin1_b: load_tensor(&mlp_lin1_b)?,
                    mlp_lin2_w: load_tensor(&mlp_lin2_w)?,
                    mlp_lin2_b: load_tensor(&mlp_lin2_b)?,
                    norm3_w: load_tensor(&norm3_w)?,
                    norm3_b: load_tensor(&norm3_b)?,
                    norm4_w: load_tensor(&norm4_w)?,
                    norm4_b: load_tensor(&norm4_b)?,
                    cross_attn_img_to_token,
                };

                transformer_layers.push(layer);
            }

            let prefix = "mask_decoder.transformer";
            let transformer_final_attn_token_to_img_q_w =
                format!("{prefix}.final_attn_token_to_image.q_proj.weight");
            let transformer_final_attn_token_to_img_q_b =
                format!("{prefix}.final_attn_token_to_image.q_proj.bias");
            let transformer_final_attn_token_to_img_k_w =
                format!("{prefix}.final_attn_token_to_image.k_proj.weight");
            let transformer_final_attn_token_to_img_k_b =
                format!("{prefix}.final_attn_token_to_image.k_proj.bias");
            let transformer_final_attn_token_to_img_v_w =
                format!("{prefix}.final_attn_token_to_image.v_proj.weight");
            let transformer_final_attn_token_to_img_v_b =
                format!("{prefix}.final_attn_token_to_image.v_proj.bias");
            let transformer_final_attn_token_to_img_out_w =
                format!("{prefix}.final_attn_token_to_image.out_proj.weight");
            let transformer_final_attn_token_to_img_out_b =
                format!("{prefix}.final_attn_token_to_image.out_proj.bias");

            let transformer_final_attn_token_to_img = SamLayerDecTransformerAttn {
                q_w: load_tensor(&transformer_final_attn_token_to_img_q_w)?,
                q_b: load_tensor(&transformer_final_attn_token_to_img_q_b)?,
                k_w: load_tensor(&transformer_final_attn_token_to_img_k_w)?,
                k_b: load_tensor(&transformer_final_attn_token_to_img_k_b)?,
                v_w: load_tensor(&transformer_final_attn_token_to_img_v_w)?,
                v_b: load_tensor(&transformer_final_attn_token_to_img_v_b)?,
                out_w: load_tensor(&transformer_final_attn_token_to_img_out_w)?,
                out_b: load_tensor(&transformer_final_attn_token_to_img_out_b)?,
            };

            let transformer_norm_final_w = format!("{prefix}.layer_norm_final_attn.weight");
            let transformer_norm_final_b = format!("{prefix}.layer_norm_final_attn.bias");

            let output_upscaling_0_w = "mask_decoder.upscale_conv1.weight";
            let output_upscaling_0_b = "mask_decoder.upscale_conv1.bias";
            let output_upscaling_1_w = "mask_decoder.upscale_layer_norm.weight";
            let output_upscaling_1_b = "mask_decoder.upscale_layer_norm.bias";
            let output_upscaling_3_w = "mask_decoder.upscale_conv2.weight";
            let output_upscaling_3_b = "mask_decoder.upscale_conv2.bias";

            let mut output_hypernet_mlps = vec![];
            let n_hyperned_mlps_count = 4;
            for i in 0..n_hyperned_mlps_count {
                let prefix = format!("mask_decoder.output_hypernetworks_mlps.{i}");
                let w_0 = format!("{prefix}.layers.0.weight");
                let b_0 = format!("{prefix}.layers.0.bias");
                let w_1 = format!("{prefix}.proj_in.weight");
                let b_1 = format!("{prefix}.proj_in.bias");
                let w_2 = format!("{prefix}.proj_out.weight");
                let b_2 = format!("{prefix}.proj_out.bias");

                let layer = SamLayerDecOutputHypernetMlps {
                    w_0: load_tensor(&w_0)?,
                    b_0: load_tensor(&b_0)?,
                    w_1: load_tensor(&w_1)?,
                    b_1: load_tensor(&b_1)?,
                    w_2: load_tensor(&w_2)?,
                    b_2: load_tensor(&b_2)?,
                };
                output_hypernet_mlps.push(layer);
            }

            let prefix = "mask_decoder.iou_prediction_head";
            let iou_prediction_head_0_w = format!("{prefix}.layers.0.weight");
            let iou_prediction_head_0_b = format!("{prefix}.layers.0.bias");
            let iou_prediction_head_1_w = format!("{prefix}.proj_in.weight");
            let iou_prediction_head_1_b = format!("{prefix}.proj_in.bias");
            let iou_prediction_head_2_w = format!("{prefix}.proj_out.weight");
            let iou_prediction_head_2_b = format!("{prefix}.proj_out.bias");

            let iou_token_w = "mask_decoder.iou_token.weight";
            let mask_tokens_w = "mask_decoder.mask_tokens.weight";

            SamDecoderMask {
                transformer_layers,
                transformer_final_attn_token_to_img,
                transformer_norm_final_w: load_tensor(&transformer_norm_final_w)?,
                transformer_norm_final_b: load_tensor(&transformer_norm_final_b)?,
                output_upscaling_0_w: load_tensor(output_upscaling_0_w)?,
                output_upscaling_0_b: load_tensor(output_upscaling_0_b)?,
                output_upscaling_1_w: load_tensor(output_upscaling_1_w)?,
                output_upscaling_1_b: load_tensor(output_upscaling_1_b)?,
                output_upscaling_3_w: load_tensor(output_upscaling_3_w)?,
                output_upscaling_3_b: load_tensor(output_upscaling_3_b)?,
                output_hypernet_mlps,
                iou_prediction_head_0_w: load_tensor(&iou_prediction_head_0_w)?,
                iou_prediction_head_0_b: load_tensor(&iou_prediction_head_0_b)?,
                iou_prediction_head_1_w: load_tensor(&iou_prediction_head_1_w)?,
                iou_prediction_head_1_b: load_tensor(&iou_prediction_head_1_b)?,
                iou_prediction_head_2_w: load_tensor(&iou_prediction_head_2_w)?,
                iou_prediction_head_2_b: load_tensor(&iou_prediction_head_2_b)?,
                iou_token_w: load_tensor(iou_token_w)?,
                mask_tokens_w: load_tensor(mask_tokens_w)?,
            }
        };

        Ok(Self {
            hparams,
            enc_img,
            enc_prompt,
            dec,
        })
    }

    pub fn from_gguf(backend: &GpuBackend, gguf: &Gguf) -> anyhow::Result<Self> {
        let storage = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
        let load_tensor = |name: &str| {
            if !gguf.tensors.contains_key(name) {
                println!("Key not found: {name}");
            }
            let t = &gguf.tensors[name];

            let dims = t.dimensions.map(|dim| dim as u32);
            let t = TensorBuilder::tensor(&dims[..t.rank as usize], storage)
                .build_init(backend, t.data.as_f32().unwrap())?;
            println!(
                "Loaded GG tensor: {} {:?} (rank: {})",
                name,
                t.layout(),
                t.rank()
            );
            Ok::<_, anyhow::Error>(t)
        };

        /*
         * hparams
         */
        let hparams = {
            let n_enc_state = gguf.metadata["n_enc_state"].unwrap_u32();
            let n_enc_layer = gguf.metadata["n_enc_layers"].unwrap_u32();
            let n_enc_head = gguf.metadata["n_enc_heads"].unwrap_u32();
            let n_enc_out_chans = gguf.metadata["n_enc_out_chans"].unwrap_u32();
            let n_pt_embd = gguf.metadata["n_pt_embd"].unwrap_u32();
            let ftype = gguf.metadata["ftype"].unwrap_u32();

            SamHParams {
                n_enc_state,
                n_enc_layer,
                n_enc_head,
                n_enc_out_chans,
                n_pt_embd,
                ftype,
                ..Default::default()
            }
        };

        /*
         * Image encoder.
         */
        let enc_img = {
            let pe = load_tensor("image_encoder.pos_embed")?;
            let proj_w = load_tensor("image_encoder.patch_embed.proj.weight")?;
            let proj_b = load_tensor("image_encoder.patch_embed.proj.bias")?;
            let neck_conv_0 = load_tensor("image_encoder.neck.0.weight")?;
            let neck_conv_1 = load_tensor("image_encoder.neck.2.weight")?;
            let neck_norm_0_w = load_tensor("image_encoder.neck.1.weight")?;
            let neck_norm_0_b = load_tensor("image_encoder.neck.1.bias")?;
            let neck_norm_1_w = load_tensor("image_encoder.neck.3.weight")?;
            let neck_norm_1_b = load_tensor("image_encoder.neck.3.bias")?;

            let mut layers = vec![];
            for i in 0..hparams.n_enc_layer {
                let norm1_w = format!("image_encoder.blocks.{i}.norm1.weight");
                let norm1_b = format!("image_encoder.blocks.{i}.norm1.bias");
                let rel_pos_w = format!("image_encoder.blocks.{i}.attn.rel_pos_w");
                let rel_pos_h = format!("image_encoder.blocks.{i}.attn.rel_pos_h");
                let qkv_w = format!("image_encoder.blocks.{i}.attn.qkv.weight");
                let qkv_b = format!("image_encoder.blocks.{i}.attn.qkv.bias");
                let proj_w = format!("image_encoder.blocks.{i}.attn.proj.weight");
                let proj_b = format!("image_encoder.blocks.{i}.attn.proj.bias");
                let norm2_w = format!("image_encoder.blocks.{i}.norm2.weight");
                let norm2_b = format!("image_encoder.blocks.{i}.norm2.bias");
                let mlp_lin1_w = format!("image_encoder.blocks.{i}.mlp.lin1.weight");
                let mlp_lin1_b = format!("image_encoder.blocks.{i}.mlp.lin1.bias");
                let mlp_lin2_w = format!("image_encoder.blocks.{i}.mlp.lin2.weight");
                let mlp_lin2_b = format!("image_encoder.blocks.{i}.mlp.lin2.bias");
                let layer = SamLayerEnc {
                    norm1_w: load_tensor(&norm1_w)?,
                    norm1_b: load_tensor(&norm1_b)?,
                    rel_pos_w: load_tensor(&rel_pos_w)?,
                    rel_pos_h: load_tensor(&rel_pos_h)?,
                    qkv_w: load_tensor(&qkv_w)?,
                    qkv_b: load_tensor(&qkv_b)?,
                    proj_w: load_tensor(&proj_w)?,
                    proj_b: load_tensor(&proj_b)?,
                    norm2_w: load_tensor(&norm2_w)?,
                    norm2_b: load_tensor(&norm2_b)?,
                    mlp_lin1_w: load_tensor(&mlp_lin1_w)?,
                    mlp_lin1_b: load_tensor(&mlp_lin1_b)?,
                    mlp_lin2_w: load_tensor(&mlp_lin2_w)?,
                    mlp_lin2_b: load_tensor(&mlp_lin2_b)?,
                };
                layers.push(layer);
            }

            SamEncoderImage {
                pe,
                proj_w,
                proj_b,
                neck_conv_0,
                neck_norm_0_w,
                neck_norm_0_b,
                neck_conv_1,
                neck_norm_1_w,
                neck_norm_1_b,
                layers,
            }
        };

        /*
         * Prompt encoder.
         */
        let enc_prompt = {
            let pe = load_tensor("prompt_encoder.pe_layer.positional_encoding_gaussian_matrix")?;
            let not_a_pt_embd_w = load_tensor("prompt_encoder.not_a_point_embed.weight")?;
            let no_mask_embd_w = load_tensor("prompt_encoder.no_mask_embed.weight")?;

            let mut pt_embd = vec![];
            for i in 0..hparams.n_pt_embd {
                let weight = format!("prompt_encoder.point_embeddings.{i}.weight");
                pt_embd.push(load_tensor(&weight)?);
            }

            SamEncoderPrompt {
                pe,
                not_a_pt_embd_w,
                pt_embd,
                no_mask_embd_w,
            }
        };

        /*
         * Mask decoder.
         */
        let dec = {
            let mut transformer_layers = vec![];
            let tfm_layers_count = 2;
            for i in 0..tfm_layers_count {
                let prefix = format!("mask_decoder.transformer.layers.{i}");
                let self_attn_q_w = format!("{prefix}.self_attn.q_proj.weight");
                let self_attn_q_b = format!("{prefix}.self_attn.q_proj.bias");
                let self_attn_k_w = format!("{prefix}.self_attn.k_proj.weight");
                let self_attn_k_b = format!("{prefix}.self_attn.k_proj.bias");
                let self_attn_v_w = format!("{prefix}.self_attn.v_proj.weight");
                let self_attn_v_b = format!("{prefix}.self_attn.v_proj.bias");
                let self_attn_out_w = format!("{prefix}.self_attn.out_proj.weight");
                let self_attn_out_b = format!("{prefix}.self_attn.out_proj.bias");

                let norm1_w = format!("{prefix}.norm1.weight");
                let norm1_b = format!("{prefix}.norm1.bias");

                let cross_attn_token_to_img_q_w =
                    format!("{prefix}.cross_attn_token_to_image.q_proj.weight");
                let cross_attn_token_to_img_q_b =
                    format!("{prefix}.cross_attn_token_to_image.q_proj.bias");
                let cross_attn_token_to_img_k_w =
                    format!("{prefix}.cross_attn_token_to_image.k_proj.weight");
                let cross_attn_token_to_img_k_b =
                    format!("{prefix}.cross_attn_token_to_image.k_proj.bias");
                let cross_attn_token_to_img_v_w =
                    format!("{prefix}.cross_attn_token_to_image.v_proj.weight");
                let cross_attn_token_to_img_v_b =
                    format!("{prefix}.cross_attn_token_to_image.v_proj.bias");
                let cross_attn_token_to_img_out_w =
                    format!("{prefix}.cross_attn_token_to_image.out_proj.weight");
                let cross_attn_token_to_img_out_b =
                    format!("{prefix}.cross_attn_token_to_image.out_proj.bias");

                let norm2_w = format!("{prefix}.norm2.weight");
                let norm2_b = format!("{prefix}.norm2.bias");

                let mlp_lin1_w = format!("{prefix}.mlp.lin1.weight");
                let mlp_lin1_b = format!("{prefix}.mlp.lin1.bias");
                let mlp_lin2_w = format!("{prefix}.mlp.lin2.weight");
                let mlp_lin2_b = format!("{prefix}.mlp.lin2.bias");

                let norm3_w = format!("{prefix}.norm3.weight");
                let norm3_b = format!("{prefix}.norm3.bias");
                let norm4_w = format!("{prefix}.norm4.weight");
                let norm4_b = format!("{prefix}.norm4.bias");

                let cross_attn_img_to_token_q_w =
                    format!("{prefix}.cross_attn_image_to_token.q_proj.weight");
                let cross_attn_img_to_token_q_b =
                    format!("{prefix}.cross_attn_image_to_token.q_proj.bias");
                let cross_attn_img_to_token_k_w =
                    format!("{prefix}.cross_attn_image_to_token.k_proj.weight");
                let cross_attn_img_to_token_k_b =
                    format!("{prefix}.cross_attn_image_to_token.k_proj.bias");
                let cross_attn_img_to_token_v_w =
                    format!("{prefix}.cross_attn_image_to_token.v_proj.weight");
                let cross_attn_img_to_token_v_b =
                    format!("{prefix}.cross_attn_image_to_token.v_proj.bias");
                let cross_attn_img_to_token_out_w =
                    format!("{prefix}.cross_attn_image_to_token.out_proj.weight");
                let cross_attn_img_to_token_out_b =
                    format!("{prefix}.cross_attn_image_to_token.out_proj.bias");

                let self_attn = SamLayerDecTransformerAttn {
                    q_w: load_tensor(&self_attn_q_w)?,
                    q_b: load_tensor(&self_attn_q_b)?,
                    k_w: load_tensor(&self_attn_k_w)?,
                    k_b: load_tensor(&self_attn_k_b)?,
                    v_w: load_tensor(&self_attn_v_w)?,
                    v_b: load_tensor(&self_attn_v_b)?,
                    out_w: load_tensor(&self_attn_out_w)?,
                    out_b: load_tensor(&self_attn_out_b)?,
                };
                let cross_attn_token_to_img = SamLayerDecTransformerAttn {
                    q_w: load_tensor(&cross_attn_token_to_img_q_w)?,
                    q_b: load_tensor(&cross_attn_token_to_img_q_b)?,
                    k_w: load_tensor(&cross_attn_token_to_img_k_w)?,
                    k_b: load_tensor(&cross_attn_token_to_img_k_b)?,
                    v_w: load_tensor(&cross_attn_token_to_img_v_w)?,
                    v_b: load_tensor(&cross_attn_token_to_img_v_b)?,
                    out_w: load_tensor(&cross_attn_token_to_img_out_w)?,
                    out_b: load_tensor(&cross_attn_token_to_img_out_b)?,
                };
                let cross_attn_img_to_token = SamLayerDecTransformerAttn {
                    q_w: load_tensor(&cross_attn_img_to_token_q_w)?,
                    q_b: load_tensor(&cross_attn_img_to_token_q_b)?,
                    k_w: load_tensor(&cross_attn_img_to_token_k_w)?,
                    k_b: load_tensor(&cross_attn_img_to_token_k_b)?,
                    v_w: load_tensor(&cross_attn_img_to_token_v_w)?,
                    v_b: load_tensor(&cross_attn_img_to_token_v_b)?,
                    out_w: load_tensor(&cross_attn_img_to_token_out_w)?,
                    out_b: load_tensor(&cross_attn_img_to_token_out_b)?,
                };
                let layer = SamLayerDecTransformer {
                    self_attn,
                    norm1_w: load_tensor(&norm1_w)?,
                    norm1_b: load_tensor(&norm1_b)?,
                    cross_attn_token_to_img,
                    norm2_w: load_tensor(&norm2_w)?,
                    norm2_b: load_tensor(&norm2_b)?,
                    mlp_lin1_w: load_tensor(&mlp_lin1_w)?,
                    mlp_lin1_b: load_tensor(&mlp_lin1_b)?,
                    mlp_lin2_w: load_tensor(&mlp_lin2_w)?,
                    mlp_lin2_b: load_tensor(&mlp_lin2_b)?,
                    norm3_w: load_tensor(&norm3_w)?,
                    norm3_b: load_tensor(&norm3_b)?,
                    norm4_w: load_tensor(&norm4_w)?,
                    norm4_b: load_tensor(&norm4_b)?,
                    cross_attn_img_to_token,
                };

                transformer_layers.push(layer);
            }

            let prefix = "mask_decoder.transformer";
            let transformer_final_attn_token_to_img_q_w =
                format!("{prefix}.final_attn_token_to_image.q_proj.weight");
            let transformer_final_attn_token_to_img_q_b =
                format!("{prefix}.final_attn_token_to_image.q_proj.bias");
            let transformer_final_attn_token_to_img_k_w =
                format!("{prefix}.final_attn_token_to_image.k_proj.weight");
            let transformer_final_attn_token_to_img_k_b =
                format!("{prefix}.final_attn_token_to_image.k_proj.bias");
            let transformer_final_attn_token_to_img_v_w =
                format!("{prefix}.final_attn_token_to_image.v_proj.weight");
            let transformer_final_attn_token_to_img_v_b =
                format!("{prefix}.final_attn_token_to_image.v_proj.bias");
            let transformer_final_attn_token_to_img_out_w =
                format!("{prefix}.final_attn_token_to_image.out_proj.weight");
            let transformer_final_attn_token_to_img_out_b =
                format!("{prefix}.final_attn_token_to_image.out_proj.bias");

            let transformer_final_attn_token_to_img = SamLayerDecTransformerAttn {
                q_w: load_tensor(&transformer_final_attn_token_to_img_q_w)?,
                q_b: load_tensor(&transformer_final_attn_token_to_img_q_b)?,
                k_w: load_tensor(&transformer_final_attn_token_to_img_k_w)?,
                k_b: load_tensor(&transformer_final_attn_token_to_img_k_b)?,
                v_w: load_tensor(&transformer_final_attn_token_to_img_v_w)?,
                v_b: load_tensor(&transformer_final_attn_token_to_img_v_b)?,
                out_w: load_tensor(&transformer_final_attn_token_to_img_out_w)?,
                out_b: load_tensor(&transformer_final_attn_token_to_img_out_b)?,
            };

            let transformer_norm_final_w = format!("{prefix}.norm_final_attn.weight");
            let transformer_norm_final_b = format!("{prefix}.norm_final_attn.bias");

            let prefix = "mask_decoder.output_upscaling";
            let output_upscaling_0_w = format!("{prefix}.0.weight");
            let output_upscaling_0_b = format!("{prefix}.0.bias");
            let output_upscaling_1_w = format!("{prefix}.1.weight");
            let output_upscaling_1_b = format!("{prefix}.1.bias");
            let output_upscaling_3_w = format!("{prefix}.3.weight");
            let output_upscaling_3_b = format!("{prefix}.3.bias");

            let mut output_hypernet_mlps = vec![];
            let n_hyperned_mlps_count = 4;
            for i in 0..n_hyperned_mlps_count {
                let prefix = format!("mask_decoder.output_hypernetworks_mlps.{i}");
                let w_0 = format!("{prefix}.layers.0.weight");
                let b_0 = format!("{prefix}.layers.0.bias");
                let w_1 = format!("{prefix}.layers.1.weight");
                let b_1 = format!("{prefix}.layers.1.bias");
                let w_2 = format!("{prefix}.layers.2.weight");
                let b_2 = format!("{prefix}.layers.2.bias");

                let layer = SamLayerDecOutputHypernetMlps {
                    w_0: load_tensor(&w_0)?,
                    b_0: load_tensor(&b_0)?,
                    w_1: load_tensor(&w_1)?,
                    b_1: load_tensor(&b_1)?,
                    w_2: load_tensor(&w_2)?,
                    b_2: load_tensor(&b_2)?,
                };
                output_hypernet_mlps.push(layer);
            }

            let prefix = "mask_decoder.iou_prediction_head.layers";
            let iou_prediction_head_0_w = format!("{prefix}.0.weight");
            let iou_prediction_head_0_b = format!("{prefix}.0.bias");
            let iou_prediction_head_1_w = format!("{prefix}.1.weight");
            let iou_prediction_head_1_b = format!("{prefix}.1.bias");
            let iou_prediction_head_2_w = format!("{prefix}.2.weight");
            let iou_prediction_head_2_b = format!("{prefix}.2.bias");

            let iou_token_w = "mask_decoder.iou_token.weight";
            let mask_tokens_w = "mask_decoder.mask_tokens.weight";

            SamDecoderMask {
                transformer_layers,
                transformer_final_attn_token_to_img,
                transformer_norm_final_w: load_tensor(&transformer_norm_final_w)?,
                transformer_norm_final_b: load_tensor(&transformer_norm_final_b)?,
                output_upscaling_0_w: load_tensor(&output_upscaling_0_w)?,
                output_upscaling_0_b: load_tensor(&output_upscaling_0_b)?,
                output_upscaling_1_w: load_tensor(&output_upscaling_1_w)?,
                output_upscaling_1_b: load_tensor(&output_upscaling_1_b)?,
                output_upscaling_3_w: load_tensor(&output_upscaling_3_w)?,
                output_upscaling_3_b: load_tensor(&output_upscaling_3_b)?,
                output_hypernet_mlps,
                iou_prediction_head_0_w: load_tensor(&iou_prediction_head_0_w)?,
                iou_prediction_head_0_b: load_tensor(&iou_prediction_head_0_b)?,
                iou_prediction_head_1_w: load_tensor(&iou_prediction_head_1_w)?,
                iou_prediction_head_1_b: load_tensor(&iou_prediction_head_1_b)?,
                iou_prediction_head_2_w: load_tensor(&iou_prediction_head_2_w)?,
                iou_prediction_head_2_b: load_tensor(&iou_prediction_head_2_b)?,
                iou_token_w: load_tensor(iou_token_w)?,
                mask_tokens_w: load_tensor(mask_tokens_w)?,
            }
        };

        Ok(Self {
            hparams,
            enc_img,
            enc_prompt,
            dec,
        })
    }
}

pub struct SamState {
    pub embd_img: Box<Tensor<f32>>,
    pub pe_img: Box<Tensor<f32>>,
    // ctx_img: ggml_context,
    pub low_res_masks: Box<Tensor<f32>>,
    pub iou_predictions: Tensor<f32>,
    pub debug: Tensor<f32>,
    // ctx_masks: ggml_context,
    // allocr: ggml_allocr,
}

impl SamState {
    pub fn new(backend: &GpuBackend) -> anyhow::Result<Self> {
        let storage = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
        Ok(SamState {
            embd_img: Box::new(Tensor::vector(backend, [], storage)?),
            pe_img: Box::new(Tensor::vector(backend, [], storage)?),
            low_res_masks: Box::new(Tensor::vector(backend, [], storage)?),
            iou_predictions: Tensor::matrix_uninit(backend, 1, 3, storage)?,
            debug: Tensor::vector(backend, [], storage)?,
        })
    }
}

// TODO: custom shader: ggml_sam_sin
// TODO: custom shader: ggml_sam_cos

pub fn sam_image_preprocess(img: &DMatrix<[u8; 3]>) -> DMatrix<[f32; 3]> {
    let nx = img.nrows();
    let ny = img.ncols();
    let nx2 = 1024;
    let ny2 = 1024;
    let mut res = DMatrix::repeat(nx2, ny2, [0.0; 3]);

    let scale = nx.max(ny) as f32 / 1024.0;
    let nx3 = (nx as f32 / scale + 0.5) as usize;
    let ny3 = (ny as f32 / scale + 0.5) as usize;
    let m3 = [123.675, 116.280, 103.530];
    let s3 = [58.395, 57.120, 57.375];

    for y in 0..ny3 {
        for x in 0..nx3 {
            for c in 0..3 {
                // linear interpolation
                let sx = (x as f32 + 0.5) * scale - 0.5;
                let sy = (y as f32 + 0.5) * scale - 0.5;
                let x0 = sx.floor().max(0.0) as usize;
                let y0 = sy.floor().max(0.0) as usize;
                let x1 = (x0 + 1).min(nx - 1);
                let y1 = (y0 + 1).min(ny - 1);

                let dx = sx - x0 as f32;
                let dy = sy - y0 as f32;

                let j00 = y0 * nx + x0;
                let j01 = y0 * nx + x1;
                let j10 = y1 * nx + x0;
                let j11 = y1 * nx + x1;

                let v00 = img[j00][c];
                let v01 = img[j01][c];
                let v10 = img[j10][c];
                let v11 = img[j11][c];

                let v0 = v00 as f32 * (1.0 - dx) + v01 as f32 * dx;
                let v1 = v10 as f32 * (1.0 - dx) + v11 as f32 * dx;

                let v = v0 * (1.0 - dy) + v1 * dy;

                let v2: u8 = v.round().clamp(0.0, 255.0) as u8;

                let i = y * nx3 + x;

                res[i][c] = (v2 as f32 - m3[c]) / s3[c];
            }
        }
    }

    res
}
