use crate::gguf::read_array_unaligned;
use crate::quantization::{BlockBF16, BlockF16};
use crate::quantized_matrix::GpuQuantTensor;
use khal::backend::{GpuBackend, GpuBackendError};
use khal::BufferUsages;
use safetensors::tensor::TensorView;
use safetensors::Dtype;
use vortx::tensor::{Tensor, TensorBuilder};

pub trait SafeTensorExt {
    fn to_gpu_tensor_f32_with_usage(
        &self,
        backend: &GpuBackend,
        usage: BufferUsages,
        debug_print: bool,
    ) -> Result<Tensor<f32>, GpuBackendError>;
    fn to_gpu_tensor_with_usage(
        &self,
        backend: &GpuBackend,
        usage: BufferUsages,
        debug_print: bool,
    ) -> Result<GpuQuantTensor, GpuBackendError>;

    fn to_gpu_tensor_f32(&self, backend: &GpuBackend) -> Result<Tensor<f32>, GpuBackendError> {
        self.to_gpu_tensor_f32_with_usage(backend, BufferUsages::STORAGE, false)
    }
    fn to_gpu_tensor(&self, backend: &GpuBackend) -> Result<GpuQuantTensor, GpuBackendError> {
        self.to_gpu_tensor_with_usage(backend, BufferUsages::STORAGE, false)
    }

    #[allow(dead_code)]
    fn to_gpu_tensor_print(
        &self,
        backend: &GpuBackend,
        debug_print: bool,
    ) -> Result<GpuQuantTensor, GpuBackendError> {
        self.to_gpu_tensor_with_usage(backend, BufferUsages::STORAGE, debug_print)
    }
}

impl<'data> SafeTensorExt for TensorView<'data> {
    fn to_gpu_tensor_f32_with_usage(
        &self,
        backend: &GpuBackend,
        usage: BufferUsages,
        debug_print: bool,
    ) -> Result<Tensor<f32>, GpuBackendError> {
        let safetensor_shape = self.shape();
        assert!(safetensor_shape.len() <= 4);
        let mut shape = [1; 4];
        for k in 0..safetensor_shape.len() {
            shape[k] = safetensor_shape[k] as u32;
        }
        let len = shape[0] * shape[1] * shape[2] * shape[3];
        let rank = safetensor_shape.len();

        let mat = match self.dtype() {
            Dtype::F32 => {
                let data: Vec<f32> =
                    read_array_unaligned(len as usize, self.data(), &mut 0).unwrap();
                TensorBuilder::tensor(&shape[..rank], usage).build_init(backend, &data)?
            }
            Dtype::F16 => {
                // TODO PERF: add actual f16 support to the gpu shaders.
                let data: Vec<BlockF16> =
                    read_array_unaligned(len as usize, self.data(), &mut 0).unwrap();
                let dequantized: Vec<f32> = data.iter().map(|val| val.dequantize()).collect();
                TensorBuilder::tensor(&shape[..rank], usage).build_init(backend, &dequantized)?
            }
            Dtype::BF16 => {
                // TODO PERF: add actual bf16 support to the gpu shaders.
                let data: Vec<BlockBF16> =
                    read_array_unaligned(len as usize, self.data(), &mut 0).unwrap();
                let dequantized: Vec<f32> = data.iter().map(|val| val.dequantize()).collect();
                if debug_print {
                    println!("Tensor {:?} (st): {:?}", shape, &dequantized[..10]);
                }
                TensorBuilder::tensor(&shape[..rank], usage).build_init(backend, &dequantized)?
            }
            _ => todo!("dtype {:?} not supported yet", self.dtype()),
        };
        Ok(mat)
    }

    fn to_gpu_tensor_with_usage(
        &self,
        backend: &GpuBackend,
        usage: BufferUsages,
        debug_print: bool,
    ) -> Result<GpuQuantTensor, GpuBackendError> {
        // TODO PERF: add actual bf16 support to the gpu shaders.
        self.to_gpu_tensor_f32_with_usage(backend, usage, debug_print)
            .map(GpuQuantTensor::F32)
    }
}
