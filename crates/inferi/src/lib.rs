#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::result_large_err)]

use khal::re_exports::include_dir::{include_dir, Dir};

/// Embedded SPIR-V shader directory.
pub static SPIRV_DIR: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/shaders-spirv");

pub mod context;
pub mod gguf;
pub mod models;
#[cfg(feature = "onnx")]
pub mod onnx;
pub mod ops;
pub mod quantization;
pub mod quantized_matrix;
mod safetensor;
pub mod tensor_cache;

pub mod re_exports {
    pub use khal;
    pub use safetensors;
    pub use tokenizers;
    pub use vortx;
}
