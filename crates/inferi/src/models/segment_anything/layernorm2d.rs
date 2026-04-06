use crate::context::LlmContext;
use crate::tensor_cache::CachedTensor;
use khal::backend::GpuBackendError;
use vortx::tensor::Tensor;

pub fn sam_layernorm_2d(
    ctxt: &mut LlmContext,
    layer: &Tensor<f32>,
    n_channels: u32,
    w: &Tensor<f32>,
    b: &Tensor<f32>,
    eps: f32,
) -> Result<CachedTensor<f32>, GpuBackendError> {
    // LayerNorm2d
    // normalize along channel dimension
    let layer = ctxt.contiguous(layer.permute_ggml([1, 2, 0, 3]))?;
    let layer = ctxt.layernorm(&layer, eps)?;
    let layer = layer.permute_ggml([2, 0, 1, 3]);

    let w = ctxt.repeat(w.reshape_ggml(&[1, 1, n_channels, 1]), layer)?;
    let b = ctxt.repeat(b.reshape_ggml(&[1, 1, n_channels, 1]), layer)?;
    let w_layer = ctxt.mul(&w, layer)?;
    ctxt.add(&w_layer, &b)
}
