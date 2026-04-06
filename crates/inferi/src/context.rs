use crate::ops::{
    conv_output_size, pool_output_size, BatchedMultiqueryAttention,
    BatchedMultiqueryAttentionParams, Concat, Conv2dNchw, ConvTranspose2d, FusedAttention, Gather,
    GemvQuant, GetRelPos, Im2Col, Im2ColConfig, LayerNorm, Pool2d, ReduceAxis, ReduceOp, RmsNorm,
    RmsNormConfig, RoPE, RoPEConfig, RoPEVariant, Select, Silu, SoftMax, Unary, UnaryOp, WinPart,
};
use crate::quantized_matrix::GpuQuantTensor;
use crate::tensor_cache::{CachedTensor, TensorCache, TensorKey};
use bytemuck::{AnyBitPattern, NoUninit, Pod};
use glamx::Vec4;
use khal::backend::{
    Backend, DeviceValue, Encoder, GpuBackend, GpuBackendError, GpuBuffer, GpuEncoder, GpuPass,
    MaybeSendSync,
};
use khal::{BufferUsages, Shader};
use std::sync::Arc;
use vortx::shapes::TensorLayoutBuffers;
use vortx::tensor::{AsTensorMut, AsTensorRef, Tensor, TensorBuilder};
use vortx::{BinOpOffsets, Contiguous, OpAssign, OpAssignVariant, Repeat};

/*
 * TODO:
 * - [ ] layernorm_inplace
 * - [ ] add_rel_pos_assign
 * - [ ] conv_transpose_2d_p0
 */

pub const GGML_0: usize = 1;
pub const GGML_1: usize = 0;
pub const GGML_2: usize = 2;
pub const GGML_3: usize = 3;

pub struct LlmOps {
    pub rms_norm: RmsNorm,
    pub rope: RoPE,
    pub silu: Silu,
    pub matmul: GemvQuant,
    pub soft_max: SoftMax,
    pub op_assign: OpAssign,
    pub attn: BatchedMultiqueryAttention,
    pub fused_attn: FusedAttention,
    pub layernorm: LayerNorm,
    pub contiguous: Contiguous,
    pub im2col: Im2Col,
    pub unop: Unary,
    pub repeat: Repeat,
    pub win_part: WinPart,
    pub get_rel_pos: GetRelPos,
    pub conv_transpose2d: ConvTranspose2d,
    pub select: Select,
    pub concat: Concat,
    pub gather: Gather,
    pub reduce_axis: ReduceAxis,
    pub pool2d: Pool2d,
    pub conv2d_nchw: Conv2dNchw,
}

impl LlmOps {
    pub fn new(backend: &GpuBackend) -> Result<Self, GpuBackendError> {
        Ok(Self {
            rms_norm: RmsNorm::from_backend(backend)?,
            rope: RoPE::from_backend(backend)?,
            silu: Silu::from_backend(backend)?,
            matmul: GemvQuant::from_backend(backend)?,
            soft_max: SoftMax::from_backend(backend)?,
            op_assign: OpAssign::from_backend(backend)?,
            attn: BatchedMultiqueryAttention::from_backend(backend)?,
            fused_attn: FusedAttention::from_backend(backend)?,
            layernorm: LayerNorm::from_backend(backend)?,
            contiguous: Contiguous::from_backend(backend)?,
            im2col: Im2Col::from_backend(backend)?,
            unop: Unary::from_backend(backend)?,
            repeat: Repeat::from_backend(backend)?,
            win_part: WinPart::from_backend(backend)?,
            get_rel_pos: GetRelPos::from_backend(backend)?,
            conv_transpose2d: ConvTranspose2d::from_backend(backend)?,
            select: Select::from_backend(backend)?,
            concat: Concat::from_backend(backend)?,
            gather: Gather::from_backend(backend)?,
            reduce_axis: ReduceAxis::from_backend(backend)?,
            pool2d: Pool2d::from_backend(backend)?,
            conv2d_nchw: Conv2dNchw::from_backend(backend)?,
        })
    }
}

pub struct LlmContextOwned {
    pub backend: Arc<GpuBackend>,
    pub cache: TensorCache,
    pub shapes: TensorLayoutBuffers,
    pub pass: Option<GpuPass>,
    pub encoder: Option<GpuEncoder>,
    pub ops: Arc<LlmOps>,
}

pub struct LlmContext<'a> {
    pub backend: &'a GpuBackend,
    pub cache: &'a mut TensorCache,
    pub shapes: &'a mut TensorLayoutBuffers,
    pub pass: Option<GpuPass>,
    pub encoder: Option<GpuEncoder>,
    pub ops: &'a LlmOps,
}

impl<'a> LlmContext<'a> {
    pub fn begin_submission(&mut self) {
        if self.encoder.is_some() {
            self.submit();
        }

        let mut encoder = self.backend.begin_encoding();
        let pass = encoder.begin_pass("inferi", None);
        self.pass = Some(pass);
        self.encoder = Some(encoder);
    }

    pub fn submit(&mut self) {
        drop(self.pass.take());
        if let Some(encoder) = self.encoder.take() {
            self.backend.submit(encoder).unwrap()
        }
    }

    pub fn ensure_submission(&mut self) {
        if self.encoder.is_some() {
            // We already have a working encoder/pass.
            return;
        }
        self.begin_submission();
    }
}

// TODO: code quality checklist:
//       - Take TensorMut for out tensors.
//       - Have all ops return Result.

impl<'a> LlmContext<'a> {
    pub fn tensor_uninit<T: DeviceValue + NoUninit>(
        &mut self,
        size: &[u32],
    ) -> Result<CachedTensor<T>, GpuBackendError> {
        assert!(size.len() <= 4);
        // COPY_DST is required for buffer copies (e.g., copy_buffer_to_buffer in reshape/copy operations)
        // COPY_SRC is required for reading results back to CPU
        let usage = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
        let key = TensorKey::with_type::<T>(size, usage);

        self.cache.get_or_insert(key, || {
            TensorBuilder::tensor(size, usage).build_uninit::<T>(self.backend)
        })
    }

    pub fn uniform<T: DeviceValue + Pod>(
        &mut self,
        data: T,
    ) -> Result<CachedTensor<T>, GpuBackendError> {
        self.tensor_with_usage(
            [1; 4],
            &[data],
            BufferUsages::STORAGE | BufferUsages::UNIFORM,
        )
    }

    pub fn tensor<T: DeviceValue + Pod, const DIM: usize>(
        &mut self,
        size: [u32; DIM],
        data: &[T],
    ) -> Result<CachedTensor<T>, GpuBackendError> {
        self.tensor_with_usage(size, data, BufferUsages::STORAGE)
    }

    pub fn tensor_with_usage<T: DeviceValue + Pod, const DIM: usize>(
        &mut self,
        size: [u32; DIM],
        data: &[T],
        usage: BufferUsages,
    ) -> Result<CachedTensor<T>, GpuBackendError> {
        // TODO add a cache.
        assert!(
            DIM <= 4,
            "tensors of dimensions higher than 4 are not supported."
        );
        // COPY_DST is required by the cache system for initialized tensors
        // so we can call `write_buffer` when recycling the tensor.
        // Note sure if this actually has any performance impact.
        let usage = usage | BufferUsages::COPY_DST;
        let key = TensorKey::with_type::<f32>(&size, usage);

        if let Some(mut tensor) = self.cache.get(key) {
            self.backend.write_buffer(tensor.buffer_mut(), 0, data)?;
            Ok(tensor)
        } else {
            let tensor = TensorBuilder::tensor(&size, usage).build_init(self.backend, data)?;
            Ok(self.cache.enroll(tensor, usage))
        }
    }

    pub fn write_buffer<T: DeviceValue + NoUninit>(
        &self,
        buffer: &mut GpuBuffer<T>,
        offset: u64,
        data: &[T],
    ) -> Result<(), GpuBackendError> {
        self.backend.write_buffer(buffer, offset, data)
    }
    pub async fn read_buffer<T: MaybeSendSync + DeviceValue + AnyBitPattern>(
        &mut self,
        buffer: &GpuBuffer<T>,
        data: &mut [T],
    ) -> Result<(), GpuBackendError> {
        self.submit();
        self.backend.read_buffer(buffer, data).await
    }
    pub async fn slow_read_buffer<T: MaybeSendSync + DeviceValue + AnyBitPattern>(
        &mut self,
        buffer: &GpuBuffer<T>,
        data: &mut [T],
    ) -> Result<(), GpuBackendError> {
        self.submit();
        self.backend.slow_read_buffer(buffer, data).await
    }

    pub async fn slow_read_vec<T: MaybeSendSync + DeviceValue + AnyBitPattern + Default>(
        &mut self,
        buffer: &GpuBuffer<T>,
    ) -> Result<Vec<T>, GpuBackendError> {
        self.submit();
        self.backend.slow_read_vec(buffer).await
    }

    pub fn clone(
        &mut self,
        t: impl AsTensorRef<f32>,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        self.contiguous(t)
    }

    pub fn contiguous_assign<T: DeviceValue + Pod>(
        &mut self,
        out: impl AsTensorMut<T>,
        value: impl AsTensorRef<T>,
    ) -> Result<(), GpuBackendError> {
        let value = value.as_tensor_ref();
        let offset_val = value.layout().offset;
        let offset = if offset_val % 256 != 0 {
            Some(self.tensor_with_usage([1], &[offset_val], BufferUsages::UNIFORM)?)
        } else {
            None
        };

        self.ensure_submission();
        self.ops.contiguous.launch(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            out,
            value,
            offset.as_deref(),
        )
    }

    pub fn contiguous<T: DeviceValue + Pod>(
        &mut self,
        value: impl AsTensorRef<T>,
    ) -> Result<CachedTensor<T>, GpuBackendError> {
        let value = value.as_tensor_ref();
        let mut result = self.tensor_uninit(value.layout().dims())?;
        self.contiguous_assign(&mut result, value)?;
        Ok(result)
    }

    pub fn select(
        &mut self,
        src: impl AsTensorRef<f32>,
        idx: impl AsTensorRef<u32>,
        dim: usize,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        assert_eq!(
            dim, 0,
            "selection over dimension other than 0 not supported yet"
        );
        let src = src.as_tensor_ref();
        let idx = idx.as_tensor_ref();
        let src_shape = src.layout();
        assert!(
            idx.is_contiguous(),
            "selection index buffer must be contiguous"
        );
        assert_eq!(
            idx.layout().rank,
            1,
            "the selection index buffer must be a vector"
        );
        let mut dest_size = src_shape.size;
        dest_size[0] = idx.len() as u32;
        let mut dest = self.tensor_uninit(&dest_size[..src_shape.rank as usize])?;

        self.ensure_submission();
        self.ops.select.launch(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            &mut dest,
            src,
            idx,
        )?;
        Ok(dest)
    }

    pub fn layernorm_assign(
        &mut self,
        out: impl AsTensorMut<f32>,
        value: impl AsTensorRef<f32>,
    ) -> Result<(), GpuBackendError> {
        self.ensure_submission();
        self.ops.layernorm.launch_rows(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            out,
            value,
        )
    }

    pub fn layernorm(
        &mut self,
        value: impl AsTensorRef<f32>,
        _eps: f32, // TODO: take this into account somehow
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let value = value.as_tensor_ref();
        let mut out = self.tensor_uninit(value.layout().dims())?;
        self.layernorm_assign(&mut out, value)?;
        Ok(out)
    }

    pub fn rms_norm_assign(
        &mut self,
        config: &Tensor<RmsNormConfig>,
        out: impl AsTensorMut<f32>,
        value: impl AsTensorRef<f32>,
        weight: impl AsTensorRef<f32>,
    ) -> Result<(), GpuBackendError> {
        self.ensure_submission();
        self.ops.rms_norm.launch(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            config,
            out,
            value,
            weight,
        )
    }

    pub fn rms_norm(
        &mut self,
        config: &Tensor<RmsNormConfig>,
        value: impl AsTensorRef<f32>,
        weight: impl AsTensorRef<f32>,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let value = value.as_tensor_ref();
        let weight = weight.as_tensor_ref();
        let mut out = self.tensor_uninit(value.layout().dims())?;
        self.rms_norm_assign(config, &mut out, value, weight)?;
        Ok(out)
    }

    /// Runs `matmul` following ggml’s unconventional behavior.
    ///
    /// All inputs are expected to be row-major.
    pub fn matmul_assign_ggml(
        &mut self,
        mut out: impl AsTensorMut<f32>,
        a: impl AsTensorRef<f32>,
        b: impl AsTensorRef<f32>,
    ) -> Result<(), GpuBackendError> {
        let out = out.as_tensor_mut();
        let a = a.as_tensor_ref();
        let b = b.as_tensor_ref();

        // NOTE: this will resolve as a GemvTrFast operation on (a, b).
        //       Capitalized variables are column-major representations of the input.
        // out = b * tr(a)
        // OUT = a * tr(b) // Make `out` column-major by transposing the equation.
        // OUT = tr(A) * B // Make `a` and `b` column-major.
        // OUT = gemm_tr_fast(A, B)
        self.matmul_assign(out, b, a.transpose_last_dims())
    }

    pub fn matmul_ggml(
        &mut self,
        a: impl AsTensorRef<f32>,
        b: impl AsTensorRef<f32>,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let a = a.as_tensor_ref();
        let b = b.as_tensor_ref();
        self.matmul(b, a.transpose_last_dims())
    }

    pub fn matmul_assign(
        &mut self,
        out: impl AsTensorMut<f32>,
        a: impl AsTensorRef<f32>,
        b: impl AsTensorRef<f32>,
    ) -> Result<(), GpuBackendError> {
        self.ops.matmul.gemm_f32.dispatch(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            out,
            a,
            b,
        )?;
        Ok(())
    }

    pub fn matmul(
        &mut self,
        a: impl AsTensorRef<f32>,
        b: impl AsTensorRef<f32>,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let a = a.as_tensor_ref();
        let b = b.as_tensor_ref();
        let out_rank = b.rank().max(a.rank());
        let a = a.canonicalize();
        let b = b.canonicalize();

        let [an, ac, ah, aw] = a.layout().size;
        let [bn, bc, bh, bw] = b.layout().size;

        assert_eq!(aw, bh);

        // TODO: fix calculation of output shape to account for broadcasting properly.
        let out_shape = [an.max(bn), ac.max(bc), ah, bw];
        let mut out = self.tensor_uninit(&out_shape[4 - out_rank as usize..])?;

        self.matmul_assign(&mut out, a, b)?;
        Ok(out)
    }

    pub fn matmul_quant_assign(
        &mut self,
        out: impl AsTensorMut<f32>,
        a: &GpuQuantTensor,
        b: impl AsTensorRef<f32>,
    ) -> Result<(), GpuBackendError> {
        self.ensure_submission();
        self.ops.matmul.launch(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            out,
            a,
            b,
        )
    }

    pub fn matmul_quant(
        &mut self,
        a: &GpuQuantTensor,
        b: impl AsTensorRef<f32>,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let b = b.as_tensor_ref();
        let [vrows, vcols, vmats, vcubes] = b.layout().size;
        let [mrows, mcols, mmats, mcubes] = a.layout().size;
        assert_eq!(mcols, vrows, "matmul dimension mismatch");
        assert_eq!(mmats, 1, "not supported");
        assert_ne!(mcubes, 1, "not supported");
        let out_shape = [mrows, vcols, vmats, vcubes];
        let mut out = self.tensor_uninit(&out_shape[..b.layout().rank as usize])?;

        self.matmul_quant_assign(&mut out, a, b)?;
        Ok(out)
    }

    /// Returns a tensor with the shape of `b` and a content equal to `a` repeated as many times
    /// as necessary to fill it.
    ///
    /// Panics if the shape of `b` isn’t an integer multiple of the shape of `a`.
    pub fn repeat(
        &mut self,
        a: impl AsTensorRef<f32>,
        b: impl AsTensorRef<f32>,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        // TODO: seems like ggml also keeps the same stride as `b`.
        let a = a.as_tensor_ref();
        let b = b.as_tensor_ref();
        for k in 0..4 {
            assert_eq!(b.size(k) % a.size(k), 0);
        }
        let mut out = self.tensor_uninit(b.layout().dims())?;
        self.ensure_submission();
        self.ops.repeat.launch(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            &mut out,
            a,
        )?;
        Ok(out)
    }

    pub fn add(
        &mut self,
        a: impl AsTensorRef<f32>,
        b: impl AsTensorRef<f32>,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let a = a.as_tensor_ref();
        let b = b.as_tensor_ref();
        let mut out = self.tensor_uninit(a.layout().dims())?;
        self.copy(a, &mut out)?;
        self.add_assign(&mut out, b)?;
        Ok(out)
    }

    pub fn add_assign(
        &mut self,
        in_out_a: impl AsTensorMut<f32>,
        b: impl AsTensorRef<f32>,
    ) -> Result<(), GpuBackendError> {
        self.ensure_submission();
        self.ops.op_assign.launch(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            OpAssignVariant::Add,
            in_out_a,
            b,
        )
    }

    /// Adds a bias tensor to a feature map in NCHW format.
    ///
    /// The feature map should have shape \[N, C, H, W\] and the bias can have:
    /// - Shape \[C\]: will be reshaped to \[1, C, 1, 1\] for broadcasting
    /// - Shape \[1, C, 1, 1\]: used directly
    /// - Shape \[C, 1, 1, 1\]: will be reshaped to \[1, C, 1, 1\]
    pub fn add_bias_nchw(
        &mut self,
        feature_map: impl AsTensorRef<f32>,
        bias: impl AsTensorRef<f32>,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let feature_map = feature_map.as_tensor_ref();
        let bias = bias.as_tensor_ref();

        let fm_layout = feature_map.layout();
        let fm_shape = fm_layout.dims();
        let bias_layout = bias.layout();
        let bias_shape = bias_layout.dims();

        assert!(
            fm_shape.len() == 4,
            "Feature map must be rank 4 (NCHW), got {:?}",
            fm_shape
        );

        let channels = fm_shape[1];

        // Determine the channel count from the bias
        let bias_channels = bias_shape.iter().find(|&&d| d > 1).copied().unwrap_or(1);

        assert_eq!(
            channels, bias_channels,
            "Bias channels {} must match feature map channels {}",
            bias_channels, channels
        );

        // Reshape bias to [1, C, 1, 1] for proper broadcasting
        // We use reshape since the bias tensor is typically contiguous [C]
        let bias_rank = bias_shape.len();
        let bias_view = if bias_rank == 1 {
            // [C] -> [1, C, 1, 1] using reshape
            bias.reshape(&[1, channels, 1, 1])
        } else if bias_rank == 2 && bias_shape[0] == 1 {
            // [1, C] -> [1, C, 1, 1]
            bias.reshape(&[1, channels, 1, 1])
        } else if bias_rank == 3 && bias_shape[1] == 1 && bias_shape[2] == 1 {
            // [C, 1, 1] -> [1, C, 1, 1] using reshape
            bias.reshape(&[1, channels, 1, 1])
        } else if bias_rank == 4 && bias_shape[0] == 1 && bias_shape[2] == 1 && bias_shape[3] == 1 {
            // Already [1, C, 1, 1]
            bias
        } else if bias_rank == 4 && bias_shape[1] == 1 && bias_shape[2] == 1 && bias_shape[3] == 1 {
            // [C, 1, 1, 1] -> [1, C, 1, 1] using reshape
            bias.reshape(&[1, channels, 1, 1])
        } else {
            panic!(
                "Unsupported bias shape {:?} for NCHW bias addition",
                bias_shape
            );
        };

        // Now do the addition with proper broadcasting
        let mut out = self.tensor_uninit(fm_shape)?;
        self.copy(feature_map, &mut out)?;
        self.add_assign(&mut out, bias_view)?;
        Ok(out)
    }

    pub fn sub(
        &mut self,
        a: impl AsTensorRef<f32>,
        b: impl AsTensorRef<f32>,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let a = a.as_tensor_ref();
        let b = b.as_tensor_ref();
        let mut out = self.tensor_uninit(a.layout().dims())?;
        self.copy(a, &mut out)?;
        self.sub_assign(&mut out, b)?;
        Ok(out)
    }

    pub fn sub_assign(
        &mut self,
        in_out_a: impl AsTensorMut<f32>,
        b: impl AsTensorRef<f32>,
    ) -> Result<(), GpuBackendError> {
        self.ensure_submission();
        self.ops.op_assign.launch(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            OpAssignVariant::Sub,
            in_out_a,
            b,
        )
    }

    pub fn mul(
        &mut self,
        a: impl AsTensorRef<f32>,
        b: impl AsTensorRef<f32>,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let a = a.as_tensor_ref();
        let b = b.as_tensor_ref();
        let mut out = self.tensor_uninit(a.layout().dims())?;
        self.copy(a, &mut out)?;
        self.mul_assign(&mut out, b)?;
        Ok(out)
    }

    pub fn mul_assign(
        &mut self,
        in_out_a: impl AsTensorMut<f32>,
        b: impl AsTensorRef<f32>,
    ) -> Result<(), GpuBackendError> {
        self.ensure_submission();
        self.ops.op_assign.launch(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            OpAssignVariant::Mul,
            in_out_a,
            b,
        )
    }

    pub fn div(
        &mut self,
        a: impl AsTensorRef<f32>,
        b: impl AsTensorRef<f32>,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let a = a.as_tensor_ref();
        let b = b.as_tensor_ref();
        let mut out = self.tensor_uninit(a.layout().dims())?;
        self.copy(a, &mut out)?;
        self.div_assign(&mut out, b)?;
        Ok(out)
    }

    pub fn div_assign(
        &mut self,
        in_out_a: impl AsTensorMut<f32>,
        b: impl AsTensorRef<f32>,
    ) -> Result<(), GpuBackendError> {
        self.ensure_submission();
        self.ops.op_assign.launch(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            OpAssignVariant::Div,
            in_out_a,
            b,
        )
    }

    pub fn rope(
        &mut self,
        variant: RoPEVariant,
        config: &Tensor<RoPEConfig>,
        in_out_q: impl AsTensorMut<f32>,
        in_out_k: impl AsTensorMut<f32>,
    ) -> Result<(), GpuBackendError> {
        self.ensure_submission();
        self.ops.rope.launch(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            variant,
            config,
            in_out_q,
            in_out_k,
        )
    }

    pub fn silu(
        &mut self,
        in_out_h1: impl AsTensorMut<f32>,
        in_h2: impl AsTensorRef<f32>,
    ) -> Result<(), GpuBackendError> {
        self.ensure_submission();
        self.ops.silu.launch(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            in_out_h1,
            in_h2,
        )
    }

    pub fn cos_inplace(&mut self, x: impl AsTensorMut<f32>) -> Result<(), GpuBackendError> {
        self.unop_inplace(UnaryOp::Cos, x)
    }

    pub fn cos(&mut self, x: impl AsTensorRef<f32>) -> Result<CachedTensor<f32>, GpuBackendError> {
        self.unop(UnaryOp::Cos, x)
    }

    pub fn sin_inplace(&mut self, x: impl AsTensorMut<f32>) -> Result<(), GpuBackendError> {
        self.unop_inplace(UnaryOp::Sin, x)
    }

    pub fn sin(&mut self, x: impl AsTensorRef<f32>) -> Result<CachedTensor<f32>, GpuBackendError> {
        self.unop(UnaryOp::Sin, x)
    }

    pub fn unop_inplace(
        &mut self,
        op: UnaryOp,
        x: impl AsTensorMut<f32>,
    ) -> Result<(), GpuBackendError> {
        self.ensure_submission();
        self.ops.unop.launch_inplace(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            op,
            x,
            None,
        )
    }

    pub fn unop(
        &mut self,
        op: UnaryOp,
        x: impl AsTensorRef<f32>,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let x = x.as_tensor_ref();
        let mut out = self.tensor_uninit(x.layout().dims())?;
        self.ensure_submission();
        self.ops.unop.launch(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            op,
            &mut out,
            x,
            None,
        )?;
        Ok(out)
    }

    /// Apply a unary operation with arguments (e.g., LeakyRelu with alpha, Clamp with min/max).
    pub fn unop_with_args(
        &mut self,
        op: UnaryOp,
        x: impl AsTensorRef<f32>,
        args: Vec4,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let args_buffer = self.uniform(args)?;
        let x = x.as_tensor_ref();
        let mut out = self.tensor_uninit(x.layout().dims())?;
        self.ensure_submission();
        self.ops.unop.launch(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            op,
            &mut out,
            x,
            Some(&args_buffer),
        )?;
        Ok(out)
    }

    pub fn gelu_inplace(&mut self, x: impl AsTensorMut<f32>) -> Result<(), GpuBackendError> {
        self.unop_inplace(UnaryOp::Gelu, x)
    }

    pub fn gelu(&mut self, x: impl AsTensorRef<f32>) -> Result<CachedTensor<f32>, GpuBackendError> {
        self.unop(UnaryOp::Gelu, x)
    }

    pub fn relu_inplace(&mut self, x: impl AsTensorMut<f32>) -> Result<(), GpuBackendError> {
        self.unop_inplace(UnaryOp::Relu, x)
    }

    pub fn relu(&mut self, x: impl AsTensorRef<f32>) -> Result<CachedTensor<f32>, GpuBackendError> {
        self.unop(UnaryOp::Relu, x)
    }

    pub fn scale_assign(
        &mut self,
        mut x: impl AsTensorMut<f32>,
        scale: f32,
    ) -> Result<(), GpuBackendError> {
        let args = self.uniform(Vec4::new(scale, 0.0, 0.0, 0.0))?;
        self.ensure_submission();
        self.ops.unop.launch_inplace(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            UnaryOp::Scale,
            x.as_tensor_mut(),
            Some(&args),
        )
    }

    pub fn scale(
        &mut self,
        x: impl AsTensorRef<f32>,
        scale: f32,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let args = self.uniform(Vec4::new(scale, 0.0, 0.0, 0.0))?;
        let x = x.as_tensor_ref();
        let mut out = self.tensor_uninit(x.layout().dims())?;
        self.ensure_submission();
        self.ops.unop.launch(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            UnaryOp::Scale,
            &mut out,
            x,
            Some(&args),
        )?;
        Ok(out)
    }

    /// Like [`Self::copy`] but with a custom buffer offset.
    ///
    /// This is to work around cases where the desired offset isn’t
    /// aligned to the hardware requirements (e.g. when targetting WebGpu).
    // TODO: find a more systematic way of dealing with this issue (this can happen
    //       for any operation, not just copying).
    pub fn copy_with_offsets(
        &mut self,
        in_: impl AsTensorRef<f32>,
        in_offset: u32,
        mut out: impl AsTensorMut<f32>,
        out_offset: u32,
    ) -> Result<(), GpuBackendError> {
        let offsets = BinOpOffsets {
            a: out_offset,
            b: in_offset,
            pad0: 0,
            pad1: 0,
        };
        let offsets = self.uniform(offsets)?;
        let out = out.as_tensor_mut();
        let in_ = in_.as_tensor_ref();
        self.ensure_submission();
        self.ops.op_assign.launch_copy_with_offsets(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            &offsets,
            out,
            in_,
        )
    }

    pub fn copy(
        &mut self,
        in_: impl AsTensorRef<f32>,
        out: impl AsTensorMut<f32>,
    ) -> Result<(), GpuBackendError> {
        self.ensure_submission();
        self.ops.op_assign.launch(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            OpAssignVariant::Copy,
            out,
            in_,
        )
    }

    /// Copy a tensor to a new owned buffer.
    pub fn copy_tensor(
        &mut self,
        in_: impl AsTensorRef<f32>,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let in_ = in_.as_tensor_ref();
        let mut out = self.tensor_uninit(in_.layout().dims())?;
        self.copy(in_, &mut out)?;
        Ok(out)
    }

    /// Copy tensor data to a new tensor with a different shape (reshape).
    ///
    /// The total number of elements must match. This does a raw buffer copy
    /// without checking shape compatibility.
    pub fn copy_tensor_with_shape(
        &mut self,
        in_: impl AsTensorRef<f32>,
        shape: &[u32],
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        use khal::backend::Encoder;

        let in_ = in_.as_tensor_ref();
        let mut out = self.tensor_uninit(shape)?;

        // Verify element count matches
        let in_len = in_.len() as usize;
        let out_len: usize = shape.iter().map(|&d| d as usize).product();
        assert_eq!(
            in_len, out_len,
            "Reshape: element count mismatch ({} vs {})",
            in_len, out_len
        );

        // End the current pass to do buffer copy
        drop(self.pass.take());

        // Do direct buffer copy using the Encoder trait
        if let Some(encoder) = self.encoder.as_mut() {
            let src_offset = in_.layout().offset as usize;
            encoder.copy_buffer_to_buffer::<f32>(
                in_.raw_buffer(),
                src_offset,
                out.tensor_mut().buffer_mut(),
                0,
                in_len,
            )?;
        }

        // Restart the pass
        if let Some(encoder) = self.encoder.as_mut() {
            self.pass = Some(encoder.begin_pass("inferi", None));
        }

        Ok(out)
    }

    pub fn win_part(
        &mut self,
        a: impl AsTensorRef<f32>,
        window_size: u32,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let a = a.as_tensor_ref();
        let w = window_size;
        assert_eq!(a.size(3), 1);

        let px = (w - a.size_ggml(1) % w) % w;
        let py = (w - a.size_ggml(2) % w) % w;

        let npx = (px + a.size_ggml(1)) / w;
        let npy = (py + a.size_ggml(2)) / w;
        let np = npx * npy;

        let res_size = [w, a.size_ggml(0), w, np];

        let mut result = self.tensor_uninit(&res_size)?;

        self.ensure_submission();
        self.ops.win_part.launch(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            &mut result,
            a,
        )?;
        Ok(result)
    }

    pub fn win_unpart(
        &mut self,
        a: impl AsTensorRef<f32>,
        w0: u32,
        h0: u32,
        window_size: u32,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let a = a.as_tensor_ref();
        let w = self.uniform(window_size)?;
        let mut result = self.tensor_uninit(&[w0, a.size_ggml(0), h0, 1])?;
        self.ensure_submission();
        self.ops.win_part.launch_unpart(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            &w,
            &mut result,
            a,
        )?;
        Ok(result)
    }

    pub fn get_rel_pos(
        &mut self,
        a: impl AsTensorRef<f32>,
        qh: u32,
        kh: u32,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let a = a.as_tensor_ref();
        let mut result = self.tensor_uninit(&[kh, a.size_ggml(0), qh, 1])?;
        self.ensure_submission();
        self.ops.get_rel_pos.launch(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            &mut result,
            a,
        )?;
        Ok(result)
    }

    pub fn add_rel_pos_assign(
        &mut self,
        dst: impl AsTensorMut<f32>,
        src1: impl AsTensorRef<f32>,
        src2: impl AsTensorRef<f32>,
    ) -> Result<(), GpuBackendError> {
        self.ensure_submission();
        self.ops.get_rel_pos.launch_add_rel_pos(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            dst,
            src1,
            src2,
        )
    }

    pub fn attn_mask(
        &mut self,
        params: &BatchedMultiqueryAttentionParams,
        params_gpu: &Tensor<BatchedMultiqueryAttentionParams>,
        attn: &mut Tensor<f32>,
    ) -> Result<(), GpuBackendError> {
        let rounded_pos = (params.pos + 1).div_ceil(4) * 4;
        let mut buf_attn = attn.buffer_mut().as_slice_mut();

        self.ensure_submission();
        self.ops.attn.mult_mask_attn.call(
            self.pass.as_mut().unwrap(),
            [params.n_heads * rounded_pos, 1, 1],
            &params_gpu.buffer().as_slice(),
            &mut buf_attn,
        )?;
        Ok(())
    }

    pub fn softmax_cols(
        &mut self,
        mut in_out_mat: impl AsTensorMut<f32>,
    ) -> Result<(), GpuBackendError> {
        let in_out_mat = in_out_mat.as_tensor_mut();
        let in_out_mat = in_out_mat.transpose_last_dims();
        self.ensure_submission();
        self.ops.soft_max.launch(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            in_out_mat,
        )
    }

    pub fn softmax_rows(
        &mut self,
        in_out_mat: impl AsTensorMut<f32>,
    ) -> Result<(), GpuBackendError> {
        self.ensure_submission();
        self.ops.soft_max.launch(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            in_out_mat,
        )
    }

    pub fn log_softmax_cols(
        &mut self,
        mut in_out_mat: impl AsTensorMut<f32>,
    ) -> Result<(), GpuBackendError> {
        let in_out_mat = in_out_mat.as_tensor_mut();
        let in_out_mat = in_out_mat.transpose_last_dims();
        self.ensure_submission();
        self.ops.soft_max.launch_log(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            in_out_mat,
        )
    }

    pub fn log_softmax_rows(
        &mut self,
        in_out_mat: impl AsTensorMut<f32>,
    ) -> Result<(), GpuBackendError> {
        self.ensure_submission();
        self.ops.soft_max.launch_log(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            in_out_mat,
        )
    }

    pub fn im2col_assign(
        &mut self,
        params: &mut Tensor<Im2ColConfig>,
        result: impl AsTensorMut<f32>,
        kernel: impl AsTensorRef<f32>, // convolution kernel
        input: impl AsTensorRef<f32>,  // data
        s0: u32,                       // stride dimension 0
        s1: u32,                       // stride dimension 1
        p0: u32,                       // padding dimension 0
        p1: u32,                       // padding dimension 1
        d0: u32,                       // dilation dimension 0
        d1: u32,                       // dilation dimension 1
        is_2d: bool, // indicates if this is a 2D convolution instead of a 1D convolution.
    ) -> Result<(), GpuBackendError> {
        self.ensure_submission();
        self.ops.im2col.launch(
            self.backend,
            self.pass.as_mut().unwrap(),
            params,
            result,
            kernel,
            input,
            s0,
            s1,
            p0,
            p1,
            d0,
            d1,
            is_2d,
        )
    }

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    // kernel: [OC，IC, KH, KW]
    // input: [N, IC, IH, IW]
    // result: [N, OH, OW, IC*KH*KW]
    pub fn im2col(
        &mut self,
        kernel: impl AsTensorRef<f32>, // convolution kernel
        input: impl AsTensorRef<f32>,  // data
        s0: u32,                       // stride dimension 0
        s1: u32,                       // stride dimension 1
        p0: u32,                       // padding dimension 0
        p1: u32,                       // padding dimension 1
        d0: u32,                       // dilation dimension 0
        d1: u32,                       // dilation dimension 1
        is_2d: bool, // indicates if this is a 2D convolution instead of a 1D convolution.
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let kernel = kernel.as_tensor_ref();
        let input = input.as_tensor_ref();
        let ksz = kernel.layout().size;
        let isz = input.layout().size;

        if is_2d {
            assert_eq!(ksz[GGML_2], isz[GGML_2]);
        } else {
            assert_eq!(ksz[GGML_1], isz[GGML_1]);
            assert_eq!(isz[GGML_3], 1);
        }

        fn conv_output_size(l: u32, r: u32, s: u32, p: u32, d: u32) -> u32 {
            (l + 2 * p - d * (r - 1) - 1) / s + 1
        }

        let oh = if is_2d {
            conv_output_size(isz[GGML_1], ksz[GGML_1], s1, p1, d1)
        } else {
            0
        };
        let ow = conv_output_size(isz[GGML_0], ksz[GGML_0], s0, p0, d0);

        let out_sz = [
            ow,
            if is_2d {
                ksz[GGML_0] * ksz[GGML_1] * ksz[GGML_2]
            } else {
                ksz[GGML_0] * ksz[GGML_1]
            },
            if is_2d { oh } else { isz[GGML_2] },
            if is_2d { isz[GGML_3] } else { 1 },
        ];

        assert!(
            !is_2d || oh > 0,
            "input too small compared to kernel: {is_2d} || {oh} > 0 failed"
        );
        assert!(
            ow > 0,
            "input too small compared to kernel: {ow} > 0 failed"
        );

        // NOTE: the `params` content is set in `Im2Col::launch`.
        let params_usage = BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_DST;
        let mut params =
            self.tensor_with_usage([1; 4], &[bytemuck::Zeroable::zeroed()], params_usage)?;

        let mut result = self.tensor(
            out_sz,
            &vec![0.0; (out_sz[0] * out_sz[1] * out_sz[2]) as usize],
        )?;
        self.im2col_assign(
            &mut params,
            &mut result,
            kernel,
            input,
            s0,
            s1,
            p0,
            p1,
            d0,
            d1,
            is_2d,
        )?;
        Ok(result)
    }

    // pub fn conv_1d(
    //     &mut self,
    //     result: impl AsTensorRef<f32>,
    //     kernel: impl AsTensorRef<f32>, // convolution kernel
    //     input: impl AsTensorRef<f32>,  // data
    //     s0: u32,                                      // stride dimension
    //     p0: u32,                                      // padding dimension
    //     d0: u32,                                      // dilation dimension
    // ) -> Result<(), GpuBackendError> {
    //     todo!()
    //     // let result = result.as_tensor_ref();
    //     // self.im2col(params, result, kernel, input, s0, 0, p0, 0, d0, 0, false);
    //     //
    //     // let rsize = result.layout().size;
    //     // let ksize = kernel.layout().size;
    //     // let lhs = result.reshape([rsize[0], rsize[1] * rsize[2]]); // [N, OL, IC * K] => [N*OL, IC * K]
    //     // let rhs = kernel.reshape([ksize[0] * ksize[1], ksize[2]]); // [OC，IC, K] => [OC, IC * K]
    //     // self.gemm(gemm_res, lhs, rhs);
    //     // gemm_res.reshape_inplace([rsize[1], ksize[2], rsize[2]]); // [N, OC, OL]
    //     // gemm_res
    // }
    //
    // // conv_1d with padding = half
    // // alias for conv_1d(a, b, s, a->ne[0]/2, d)
    // pub fn conv_1d_ph(
    //     &mut self,
    //     result: impl AsTensorRef<f32>,
    //     kernel: impl AsTensorRef<f32>, // convolution kernel
    //     input: impl AsTensorRef<f32>,  // data
    //     s0: u32,                                      // stride dimension
    //     d0: u32,                                      // dilation dimension
    // ) -> Result<(), GpuBackendError> {
    //     let kernel = kernel.as_tensor_ref();
    //     self.conv_1d(result, kernel, input, s0, kernel.layout().size[0] / 2, d0)
    // }
    //
    // // depthwise
    // // TODO: this is very likely wrong for some cases! - needs more testing
    // pub fn conv_1d_dw(
    //     &mut self,
    //     result: impl AsTensorRef<f32>,
    //     kernel: impl AsTensorRef<f32>, // convolution kernel
    //     input: impl AsTensorRef<f32>,  // data
    //     s0: u32,                                      // stride dimension
    //     p0: u32,                                      // padding dimension
    //     d0: u32,                                      // dilation dimension
    // ) -> Result<(), GpuBackendError> {
    //     todo!()
    // }
    //
    // pub fn conv_1d_dw_ph(
    //     &mut self,
    //     result: impl AsTensorRef<f32>,
    //     kernel: impl AsTensorRef<f32>, // convolution kernel
    //     input: impl AsTensorRef<f32>,  // data
    //     s0: u32,                                      // stride dimension
    //     d0: u32,                                      // padding dimension
    // ) -> Result<(), GpuBackendError> {
    //     let kernel = kernel.as_tensor_ref();
    //     self.conv_1d_dw(result, kernel, input, s0, kernel.layout().size[0] / 2, d0)
    // }
    //
    // pub fn conv_transpose_1d(
    //     &mut self,
    //     result: impl AsTensorRef<f32>,
    //     kernel: impl AsTensorRef<f32>, // convolution kernel
    //     input: impl AsTensorRef<f32>,  // data
    //     s0: u32,                                      // stride dimension
    //     p0: u32,                                      // padding dimension
    //     d0: u32,                                      // dilation dimension
    // ) -> Result<(), GpuBackendError> {
    //     todo!()
    // }

    pub fn im2col_sk_p0(
        &mut self,
        kernel: impl AsTensorRef<f32>, // convolution kernel
        input: impl AsTensorRef<f32>,  // data
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let kernel = kernel.as_tensor_ref();
        let ksize = kernel.layout().size;
        self.im2col(
            kernel,
            input,
            ksize[GGML_0],
            ksize[GGML_1],
            0,
            0,
            1,
            1,
            true,
        )
    }

    pub fn conv_2d(
        &mut self,
        kernel: impl AsTensorRef<f32>, // convolution kernel
        input: impl AsTensorRef<f32>,  // data
        s0: u32,                       // stride dimension 0
        s1: u32,                       // stride dimension 1
        p0: u32,                       // padding dimension 0
        p1: u32,                       // padding dimension 1
        d0: u32,                       // dilation dimension 0
        d1: u32,                       // dilation dimension 1
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let kernel = kernel.as_tensor_ref();
        let input = input.as_tensor_ref();
        let im2col = self.im2col(kernel, input, s0, s1, p0, p1, d0, d1, true)?; // [N, OH, OW, IC * KH * KW]

        // [N, OH, OW, IC * KH * KW] => [N*OH*OW, IC * KH * KW]
        let reshaped_im2col = im2col.reshape_ggml(&[
            im2col.size_ggml(0),
            im2col.size_ggml(3) * im2col.size_ggml(2) * im2col.size_ggml(1),
            1,
            1,
        ]);
        // [OC，IC, KH, KW] => [OC, IC * KH * KW]
        let reshaped_kernel = kernel.reshape_ggml(&[
            kernel.size_ggml(0) * kernel.size_ggml(1) * kernel.size_ggml(2),
            kernel.size_ggml(3),
            1,
            1,
        ]);
        // let reshaped_im2col = self.contiguous(reshaped_im2col)?;
        // let reshaped_kernel = self.contiguous(reshaped_kernel)?;

        let result = self.matmul_ggml(reshaped_im2col, reshaped_kernel)?;

        // reshape => [OC, N, OH, OW]
        // permute => [N, OC, OH, OW]
        self.contiguous(
            result
                .reshape_ggml(&[
                    im2col.size_ggml(1),
                    im2col.size_ggml(2),
                    im2col.size_ggml(3),
                    kernel.size_ggml(3),
                ])
                .permute_ggml([0, 1, 3, 2]),
        )
    }

    // kernel size is a->ne[0] x a->ne[1]
    // stride is equal to kernel size
    // padding is zero
    // example:
    // a:     16   16    3  768
    // b:   1024 1024    3    1
    // res:   64   64  768    1
    // used in sam
    pub fn conv_2d_sk_p0(
        &mut self,
        kernel: impl AsTensorRef<f32>, // convolution kernel
        input: impl AsTensorRef<f32>,  // data
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let kernel = kernel.as_tensor_ref();
        let ksize = kernel.layout().size;
        self.conv_2d(kernel, input, ksize[GGML_0], ksize[GGML_1], 0, 0, 1, 1)
    }

    // kernel size is a->ne[0] x a->ne[1]
    // stride is 1
    // padding is half
    // example:
    // a:      3    3    256  256
    // b:     64   64    256    1
    // res:   64   64    256    1
    // used in sam
    pub fn conv_2d_s1_ph(
        &mut self,
        kernel: impl AsTensorRef<f32>, // convolution kernel
        input: impl AsTensorRef<f32>,  // data
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let kernel = kernel.as_tensor_ref();
        let ksize = kernel.layout().size;
        self.conv_2d(
            kernel,
            input,
            1,
            1,
            ksize[GGML_0] / 2,
            ksize[GGML_1] / 2,
            1,
            1,
        )
    }

    // // depthwise
    // pub fn conv_2d_dw(
    //     &mut self,
    //     kernel: impl AsTensorRef<f32>, // convolution kernel
    //     input: impl AsTensorRef<f32>,  // data
    //     s0: u32,                                      // stride dimension 0
    //     s1: u32,                                      // stride dimension 1
    //     p0: u32,                                      // padding dimension 0
    //     p1: u32,                                      // padding dimension 1
    //     d0: u32,                                      // dilation dimension 0
    //     d1: u32,                                      // dilation dimension 1
    // ) -> Result<CachedTensor<f32>, GpuBackendError> {
    //     todo!()
    // }

    pub fn conv_transpose_2d_p0(
        &mut self,
        kernel: impl AsTensorRef<f32>, // convolution kernel
        input: impl AsTensorRef<f32>,  // data
        stride: u32,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let kernel = kernel.as_tensor_ref(); // a
        let input = input.as_tensor_ref(); // b
        assert_eq!(kernel.size_ggml(3), input.size_ggml(2));

        fn conv_transpose_output_size(ins: u32, ks: u32, s: u32, p: u32) -> u32 {
            (ins - 1) * s - 2 * p + ks
        }

        let output_sz = [
            conv_transpose_output_size(input.size_ggml(1), kernel.size_ggml(1), stride, 0),
            conv_transpose_output_size(input.size_ggml(0), kernel.size_ggml(0), stride, 0),
            kernel.size_ggml(2),
            input.size_ggml(3),
        ];
        let mut output = self.tensor_uninit(&output_sz)?;
        let mut wdata = self.tensor_uninit(&[(input.len() + kernel.len()) as u32, 1, 1, 1])?;
        let stride = self.uniform(stride)?;

        self.ensure_submission();
        self.ops.conv_transpose2d.launch_ref(
            self.backend,
            self.pass.as_mut().unwrap(),
            self.shapes,
            &stride,
            &mut output,
            kernel,
            input,
            &mut wdata,
        )?;
        Ok(output)
    }

    /// Gather elements from a tensor based on indices along a given axis.
    pub fn gather(
        &mut self,
        src: impl AsTensorRef<f32>,
        indices: impl AsTensorRef<i32>,
        axis: u32,
        output_shape: &[u32],
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let mut dest = self.tensor_uninit(output_shape)?;
        self.ensure_submission();
        self.ops.gather.launch(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            &mut dest,
            src,
            indices,
            axis,
        )?;
        Ok(dest)
    }

    /// Concatenate tensors along an axis by copying each input into the output.
    pub fn concat_copy(
        &mut self,
        dest: impl AsTensorMut<f32>,
        src: impl AsTensorRef<f32>,
        axis: u32,
        offset: u32,
    ) -> Result<(), GpuBackendError> {
        self.ensure_submission();
        self.ops.concat.launch_copy(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            dest,
            src,
            axis,
            offset,
        )
    }

    /// Reduce sum along a single axis.
    pub fn reduce_sum_axis(
        &mut self,
        src: impl AsTensorRef<f32>,
        axis: u32,
        output_shape: &[u32],
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let src_ref = src.as_tensor_ref();
        let reduce_size = src_ref.layout().size[axis as usize];
        let mut dest = self.tensor_uninit(output_shape)?;
        self.ensure_submission();
        self.ops.reduce_axis.launch(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            ReduceOp::Sum,
            &mut dest,
            src_ref,
            axis,
            reduce_size,
        )?;
        Ok(dest)
    }

    /// Reduce mean along a single axis.
    pub fn reduce_mean_axis(
        &mut self,
        src: impl AsTensorRef<f32>,
        axis: u32,
        output_shape: &[u32],
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let src_ref = src.as_tensor_ref();
        let reduce_size = src_ref.layout().size[axis as usize];
        let mut dest = self.tensor_uninit(output_shape)?;
        self.ensure_submission();
        self.ops.reduce_axis.launch(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            ReduceOp::Mean,
            &mut dest,
            src_ref,
            axis,
            reduce_size,
        )?;
        Ok(dest)
    }

    /// Reduce max along a single axis.
    pub fn reduce_max_axis(
        &mut self,
        src: impl AsTensorRef<f32>,
        axis: u32,
        output_shape: &[u32],
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let src_ref = src.as_tensor_ref();
        let reduce_size = src_ref.layout().size[axis as usize];
        let mut dest = self.tensor_uninit(output_shape)?;
        self.ensure_submission();
        self.ops.reduce_axis.launch(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            ReduceOp::Max,
            &mut dest,
            src_ref,
            axis,
            reduce_size,
        )?;
        Ok(dest)
    }

    /// Reduce min along a single axis.
    pub fn reduce_min_axis(
        &mut self,
        src: impl AsTensorRef<f32>,
        axis: u32,
        output_shape: &[u32],
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let src_ref = src.as_tensor_ref();
        let reduce_size = src_ref.layout().size[axis as usize];
        let mut dest = self.tensor_uninit(output_shape)?;
        self.ensure_submission();
        self.ops.reduce_axis.launch(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            ReduceOp::Min,
            &mut dest,
            src_ref,
            axis,
            reduce_size,
        )?;
        Ok(dest)
    }

    // =========================================================================
    // 2D Pooling operations
    // =========================================================================

    /// MaxPool2d operation (NCHW format).
    ///
    /// Input: [N, C, H, W], Output: [N, C, H_out, W_out]
    pub fn max_pool_2d(
        &mut self,
        input: impl AsTensorRef<f32>,
        kernel_h: u32,
        kernel_w: u32,
        stride_h: u32,
        stride_w: u32,
        pad_h: u32,
        pad_w: u32,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let input = input.as_tensor_ref();
        let layout = input.layout();

        // Input shape in NCHW format [N, C, H, W]
        let batch_size = layout.size[0];
        let channels = layout.size[1];
        let input_h = layout.size[2];
        let input_w = layout.size[3];

        // Compute output dimensions
        let output_h = pool_output_size(input_h, kernel_h, stride_h, pad_h, 1, false);
        let output_w = pool_output_size(input_w, kernel_w, stride_w, pad_w, 1, false);

        let output_shape = [batch_size, channels, output_h, output_w];
        let mut output = self.tensor_uninit(&output_shape)?;

        // Create params buffer
        let params_data: [u32; 16] = [
            input_h, input_w, output_h, output_w, kernel_h, kernel_w, stride_h, stride_w, pad_h,
            pad_w, channels, batch_size, 0, 0, 0, 0, // padding
        ];
        let params: CachedTensor<u32> =
            self.tensor_with_usage([16], &params_data, BufferUsages::STORAGE)?;

        self.ensure_submission();
        self.ops.pool2d.launch_max_pool(
            self.pass.as_mut().unwrap(),
            params.buffer(),
            input,
            &mut output,
        )?;

        Ok(output)
    }

    /// AvgPool2d operation (NCHW format).
    ///
    /// Input: [N, C, H, W], Output: [N, C, H_out, W_out]
    pub fn avg_pool_2d(
        &mut self,
        input: impl AsTensorRef<f32>,
        kernel_h: u32,
        kernel_w: u32,
        stride_h: u32,
        stride_w: u32,
        pad_h: u32,
        pad_w: u32,
        count_include_pad: bool,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let input = input.as_tensor_ref();
        let layout = input.layout();

        // Input shape in NCHW format [N, C, H, W]
        let batch_size = layout.size[0];
        let channels = layout.size[1];
        let input_h = layout.size[2];
        let input_w = layout.size[3];

        // Compute output dimensions
        let output_h = pool_output_size(input_h, kernel_h, stride_h, pad_h, 1, false);
        let output_w = pool_output_size(input_w, kernel_w, stride_w, pad_w, 1, false);

        let output_shape = [batch_size, channels, output_h, output_w];
        let mut output = self.tensor_uninit(&output_shape)?;

        // Create params buffer
        let params_data: [u32; 16] = [
            input_h,
            input_w,
            output_h,
            output_w,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            channels,
            batch_size,
            if count_include_pad { 1 } else { 0 },
            0,
            0,
            0,
        ];
        let params: CachedTensor<u32> =
            self.tensor_with_usage([16], &params_data, BufferUsages::STORAGE)?;

        self.ensure_submission();
        self.ops.pool2d.launch_avg_pool(
            self.pass.as_mut().unwrap(),
            params.buffer(),
            input,
            &mut output,
        )?;

        Ok(output)
    }

    /// GlobalAveragePool operation (NCHW format).
    ///
    /// Input: [N, C, H, W], Output: [N, C, 1, 1]
    pub fn global_avg_pool_2d(
        &mut self,
        input: impl AsTensorRef<f32>,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let input = input.as_tensor_ref();
        let layout = input.layout();

        // Input shape in NCHW format [N, C, H, W]
        let batch_size = layout.size[0];
        let channels = layout.size[1];
        let input_h = layout.size[2];
        let input_w = layout.size[3];

        // Output: [N, C, 1, 1] (NCHW format)
        let output_shape = [batch_size, channels, 1, 1];
        let mut output = self.tensor_uninit(&output_shape)?;

        // Create params buffer
        let params_data: [u32; 4] = [input_h, input_w, channels, batch_size];
        let params: CachedTensor<u32> =
            self.tensor_with_usage([4], &params_data, BufferUsages::STORAGE)?;

        self.ensure_submission();
        self.ops.pool2d.launch_global_avg_pool(
            self.pass.as_mut().unwrap(),
            params.buffer(),
            input,
            &mut output,
        )?;

        Ok(output)
    }

    /// GlobalMaxPool operation (NCHW format).
    ///
    /// Input: [N, C, H, W], Output: [N, C, 1, 1]
    pub fn global_max_pool_2d(
        &mut self,
        input: impl AsTensorRef<f32>,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let input = input.as_tensor_ref();
        let layout = input.layout();

        // Input shape in NCHW format [N, C, H, W]
        let batch_size = layout.size[0];
        let channels = layout.size[1];
        let input_h = layout.size[2];
        let input_w = layout.size[3];

        // Output: [N, C, 1, 1] (NCHW format)
        let output_shape = [batch_size, channels, 1, 1];
        let mut output = self.tensor_uninit(&output_shape)?;

        // Create params buffer
        let params_data: [u32; 4] = [input_h, input_w, channels, batch_size];
        let params: CachedTensor<u32> =
            self.tensor_with_usage([4], &params_data, BufferUsages::STORAGE)?;

        self.ensure_submission();
        self.ops.pool2d.launch_global_max_pool(
            self.pass.as_mut().unwrap(),
            params.buffer(),
            input,
            &mut output,
        )?;

        Ok(output)
    }

    // =========================================================================
    // NCHW-format convolution (for ONNX compatibility)
    // =========================================================================

    /// Conv2d operation with NCHW tensor format (ONNX-compatible).
    ///
    /// Input: [N, C_in, H, W]
    /// Weight: [C_out, C_in/groups, K_H, K_W]
    /// Output: [N, C_out, H_out, W_out]
    ///
    /// For standard convolution, groups=1.
    /// For depthwise convolution, groups=C_in=C_out.
    pub fn conv_2d_nchw(
        &mut self,
        input: impl AsTensorRef<f32>,
        weight: impl AsTensorRef<f32>,
        stride_h: u32,
        stride_w: u32,
        pad_h: u32,
        pad_w: u32,
        dilation_h: u32,
        dilation_w: u32,
        groups: u32,
    ) -> Result<CachedTensor<f32>, GpuBackendError> {
        let input = input.as_tensor_ref();
        let weight = weight.as_tensor_ref();

        let input_layout = input.layout();
        let weight_layout = weight.layout();

        // Input shape stored as ONNX format [N, C_in, H, W]
        // size[0]=N, size[1]=C, size[2]=H, size[3]=W
        let batch_size = input_layout.size[0];
        let in_channels = input_layout.size[1];
        let input_h = input_layout.size[2];
        let input_w = input_layout.size[3];

        // Weight shape stored as ONNX format [C_out, C_in/groups, K_H, K_W]
        let out_channels = weight_layout.size[0];
        let kernel_h = weight_layout.size[2];
        let kernel_w = weight_layout.size[3];

        // Validate groups
        assert!(
            groups > 0 && in_channels % groups == 0 && out_channels % groups == 0,
            "groups ({}) must evenly divide both in_channels ({}) and out_channels ({})",
            groups,
            in_channels,
            out_channels
        );

        // Compute output dimensions
        let output_h = conv_output_size(input_h, kernel_h, stride_h, pad_h, dilation_h);
        let output_w = conv_output_size(input_w, kernel_w, stride_w, pad_w, dilation_w);

        // Output shape stored as ONNX format [N, C_out, H_out, W_out]
        let output_shape = [batch_size, out_channels, output_h, output_w];
        let mut output = self.tensor_uninit(&output_shape)?;

        // Create params buffer
        let params_data: [u32; 16] = [
            input_h,
            input_w,
            output_h,
            output_w,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            in_channels,
            out_channels,
            batch_size,
            groups,
        ];
        let params: CachedTensor<u32> =
            self.tensor_with_usage([16], &params_data, BufferUsages::STORAGE)?;

        self.ensure_submission();
        self.ops.conv2d_nchw.launch(
            self.pass.as_mut().unwrap(),
            params.buffer(),
            input,
            weight,
            &mut output,
        )?;

        Ok(output)
    }
}
