//! Inferi shaders for rust-gpu.
//!
//! This crate contains GPU shaders for LLM inference operations, written for rust-gpu.

// Only no_std when targeting GPU (spirv/nvptx64). On CPU, we need std for generated ShaderArgs.
#![cfg_attr(any(target_arch = "spirv", target_arch = "nvptx64"), no_std)]
#![allow(clippy::too_many_arguments)]
#![allow(unexpected_cfgs)]
// Shader entry points and their constants appear dead on host but are used on GPU.
#![allow(dead_code, non_snake_case)]

pub mod batched_multiquery_attention;
pub mod concat;
pub mod conv2d;
pub mod conv_transpose_2d;
pub mod fused_attention;
pub mod gather;
pub mod gemv_quant_q4_0x2;
pub mod gemv_quant_q4_1x2;
pub mod gemv_quant_q4_k;
pub mod gemv_quant_q5_0x2;
pub mod gemv_quant_q5_1x2;
pub mod gemv_quant_q5_k;
pub mod gemv_quant_q6_kx2;
pub mod gemv_quant_q8_0x2;
pub mod gemv_quant_q8_k;
pub mod get_rel_pos;
pub mod im2col;
pub mod layernorm;
pub mod pool2d;
pub mod reduce_axis;
pub mod rms_norm;
pub mod rope;
pub mod select;
pub mod silu;
pub mod softmax;
pub mod unary;
pub mod utils;
pub mod win_part;

// Re-export shader entry points for discoverability
pub use batched_multiquery_attention::mult_mask_attn;
pub use concat::concat_copy;
pub use conv2d::conv_2d_nchw;
pub use conv_transpose_2d::{
    conv_transpose_2d, conv_transpose_2d_ref, init_dest, init_src_a, init_src_b, init_wdata,
};
pub use fused_attention::{flash_attention, fused_attention, fused_attention_online};
pub use gather::gather;
pub use get_rel_pos::{add_rel_pos_phase_a, add_rel_pos_phase_b, get_rel_pos};
pub use im2col::im2col;
pub use layernorm::{layernorm_cols, layernorm_rows};
pub use pool2d::{avg_pool_2d, global_avg_pool_2d, global_max_pool_2d, max_pool_2d};
pub use reduce_axis::{reduce_max_axis, reduce_mean_axis, reduce_min_axis, reduce_sum_axis};
pub use rms_norm::rms_norm;
pub use rope::{rope, rope_neox};
pub use select::select;
pub use silu::silu;
pub use softmax::{log_softmax, softmax};
pub use unary::*;
pub use win_part::{win_part, win_unpart};
