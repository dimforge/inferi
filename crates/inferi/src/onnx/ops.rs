//! ONNX operator dispatch to vortx/inferi implementations.

use crate::context::LlmContext;
use crate::onnx::error::OnnxError;
use crate::onnx::graph::OnnxNode;
use crate::ops::UnaryOp;
use crate::tensor_cache::CachedTensor;
use std::collections::HashMap;
use vortx::tensor::{AsTensorRef, TensorRef};

/// Execute an ONNX node, returning its output tensors.
///
/// `output_shapes` contains the pre-computed shapes for each output tensor.
pub fn dispatch_node<'a>(
    node: &OnnxNode,
    inputs: &HashMap<String, TensorRef<'a, f32>>,
    output_shapes: &[Vec<u32>],
    ctxt: &mut LlmContext<'_>,
) -> Result<Vec<CachedTensor<f32>>, OnnxError> {
    match node.op_type.as_str() {
        // Unary activation functions
        "Relu" => dispatch_unary(node, inputs, ctxt, UnaryOp::Relu),
        "Sigmoid" => dispatch_unary(node, inputs, ctxt, UnaryOp::Sigmoid),
        "Tanh" => dispatch_unary(node, inputs, ctxt, UnaryOp::Tanh),
        "Gelu" => dispatch_unary(node, inputs, ctxt, UnaryOp::Gelu),
        "Silu" => dispatch_unary(node, inputs, ctxt, UnaryOp::Silu),
        "Abs" => dispatch_unary(node, inputs, ctxt, UnaryOp::Abs),
        "Neg" => dispatch_unary(node, inputs, ctxt, UnaryOp::Neg),
        "Sqrt" => dispatch_unary(node, inputs, ctxt, UnaryOp::Sqrt),
        "Log" => dispatch_unary(node, inputs, ctxt, UnaryOp::Log),
        "Sin" => dispatch_unary(node, inputs, ctxt, UnaryOp::Sin),
        "Cos" => dispatch_unary(node, inputs, ctxt, UnaryOp::Cos),
        "Elu" => dispatch_elu(node, inputs, ctxt),
        "LeakyRelu" => dispatch_leaky_relu(node, inputs, ctxt),
        "HardSigmoid" => dispatch_unary(node, inputs, ctxt, UnaryOp::HardSigmoid),
        "Exp" => dispatch_unary(node, inputs, ctxt, UnaryOp::Exp),
        "Reciprocal" => dispatch_unary(node, inputs, ctxt, UnaryOp::Reciprocal),
        "Erf" => dispatch_unary(node, inputs, ctxt, UnaryOp::Erf),
        "Pow" => dispatch_pow(node, inputs, ctxt),

        // Clip (min/max clamping)
        "Clip" => dispatch_clip(node, inputs, ctxt),

        // Binary operations
        "Add" => dispatch_binary_add(node, inputs, ctxt),
        "Sub" => dispatch_binary_sub(node, inputs, ctxt),
        "Mul" => dispatch_binary_mul(node, inputs, ctxt),
        "Div" => dispatch_binary_div(node, inputs, ctxt),

        // Matrix operations
        "MatMul" => dispatch_matmul(node, inputs, ctxt),
        "Gemm" => dispatch_gemm(node, inputs, ctxt),

        // Normalization
        "Softmax" => dispatch_softmax(node, inputs, ctxt),
        "LayerNormalization" => dispatch_layernorm(node, inputs, ctxt),

        // Shape operations (use pre-computed output shapes)
        "Reshape" => dispatch_reshape(node, inputs, output_shapes, ctxt),
        "Transpose" => dispatch_transpose(node, inputs, ctxt),
        "Squeeze" => dispatch_squeeze(node, inputs, output_shapes, ctxt),
        "Unsqueeze" => dispatch_unsqueeze(node, inputs, output_shapes, ctxt),
        "Flatten" => dispatch_flatten(node, inputs, output_shapes, ctxt),

        // Identity / Copy / No-op operations
        "Identity" => dispatch_identity(node, inputs, ctxt),
        // Dropout is identity during inference (no training mode support)
        "Dropout" => dispatch_identity(node, inputs, ctxt),

        // Constant
        "Constant" => dispatch_constant(node, ctxt),
        "ConstantOfShape" => dispatch_constant_of_shape(node, inputs, output_shapes, ctxt),

        // Shape operation - returns the shape of a tensor
        "Shape" => dispatch_shape(node, inputs, ctxt),

        // Slice / Expand / Where
        "Slice" => dispatch_slice(node, inputs, output_shapes, ctxt),
        "Expand" => dispatch_expand(node, inputs, output_shapes, ctxt),
        "Where" => dispatch_where(node, inputs, ctxt),
        "Equal" => dispatch_equal(node, inputs, ctxt),

        // Gather / Concat
        "Gather" => dispatch_gather(node, inputs, ctxt),
        "Concat" => dispatch_concat(node, inputs, ctxt),

        // Reduce operations
        "ReduceSum" => dispatch_reduce_sum(node, inputs, ctxt),
        "ReduceMean" => dispatch_reduce_mean(node, inputs, ctxt),
        "ReduceMax" => dispatch_reduce_max(node, inputs, ctxt),
        "ReduceMin" => dispatch_reduce_min(node, inputs, ctxt),

        // Pooling operations
        "MaxPool" => dispatch_max_pool(node, inputs, ctxt),
        "AveragePool" => dispatch_avg_pool(node, inputs, ctxt),
        "GlobalAveragePool" => dispatch_global_avg_pool(node, inputs, ctxt),
        "GlobalMaxPool" => dispatch_global_max_pool(node, inputs, ctxt),

        // Conv operation
        "Conv" => dispatch_conv(node, inputs, ctxt),

        // BatchNormalization (inference mode)
        "BatchNormalization" => dispatch_batch_norm(node, inputs, ctxt),

        // TODO: implement these ops
        // "Split" => dispatch_split(node, inputs, ctxt),
        // "Slice" => dispatch_slice(node, inputs, ctxt),
        // "ConvTranspose" => dispatch_conv_transpose(node, inputs, ctxt),
        // "Cast" => dispatch_cast(node, inputs, ctxt),
        // "Where" => dispatch_where(node, inputs, ctxt),
        op => Err(OnnxError::UnsupportedOp {
            op: op.to_string(),
            node: node.name.clone(),
        }),
    }
}

/// Get an input tensor by index.
fn get_input<'a>(
    node: &OnnxNode,
    inputs: &'a HashMap<String, TensorRef<'a, f32>>,
    index: usize,
) -> Result<TensorRef<'a, f32>, OnnxError> {
    let name = node.inputs.get(index).ok_or_else(|| {
        OnnxError::MissingInput(format!(
            "Node '{}' expects input at index {}",
            node.name, index
        ))
    })?;

    if name.is_empty() {
        return Err(OnnxError::MissingInput(format!(
            "Node '{}' has empty input at index {}",
            node.name, index
        )));
    }

    inputs
        .get(name)
        .copied()
        .ok_or_else(|| OnnxError::MissingInput(name.clone()))
}

/// Get an optional input tensor by index.
fn get_optional_input<'a>(
    node: &OnnxNode,
    inputs: &'a HashMap<String, TensorRef<'a, f32>>,
    index: usize,
) -> Option<TensorRef<'a, f32>> {
    let name = node.inputs.get(index)?;
    if name.is_empty() {
        return None;
    }
    inputs.get(name).copied()
}

// =============================================================================
// Broadcasting helpers
// =============================================================================

/// Compute the broadcasted output shape for two input shapes.
/// ONNX broadcasting aligns shapes from the right, padding with 1s on the left.
fn compute_broadcast_shape(a_shape: &[u32], b_shape: &[u32]) -> Vec<u32> {
    let max_rank = a_shape.len().max(b_shape.len());
    let mut result = vec![1u32; max_rank];

    // Align from the right: position i in result corresponds to
    // position i - (max_rank - len) in the original shape
    for i in 0..max_rank {
        let a_dim = if i >= max_rank - a_shape.len() {
            a_shape[i - (max_rank - a_shape.len())]
        } else {
            1 // Implicit leading 1
        };

        let b_dim = if i >= max_rank - b_shape.len() {
            b_shape[i - (max_rank - b_shape.len())]
        } else {
            1 // Implicit leading 1
        };

        result[i] = a_dim.max(b_dim);
    }

    result
}

/// Pad a tensor's shape with leading 1s to match the target rank.
/// Returns a reshaped view if needed, otherwise returns the original tensor.
fn pad_for_broadcast<'a>(tensor: TensorRef<'a, f32>, target_rank: usize) -> TensorRef<'a, f32> {
    let layout = tensor.layout();
    let current_rank = layout.rank as usize;
    if current_rank >= target_rank {
        return tensor;
    }

    // Build new shape with leading 1s
    let current_shape = layout.dims();
    let mut new_shape = vec![1u32; target_rank];
    for i in 0..current_rank {
        new_shape[target_rank - current_rank + i] = current_shape[i];
    }

    // Reshape the tensor (metadata-only operation)
    tensor.reshape(&new_shape)
}

/// Dispatch a binary operation with proper ONNX broadcasting.
/// Creates an output tensor with the broadcast shape, broadcasts the first operand into it,
/// then applies the operation with the second operand.
fn dispatch_broadcast_binary_op<'a, F>(
    a: TensorRef<'a, f32>,
    b: TensorRef<'a, f32>,
    ctxt: &mut LlmContext<'_>,
    op: F,
) -> Result<CachedTensor<f32>, OnnxError>
where
    F: FnOnce(
        &mut LlmContext<'_>,
        &mut CachedTensor<f32>,
        TensorRef<'_, f32>,
    ) -> Result<(), khal::backend::GpuBackendError>,
{
    let a_layout = a.layout();
    let b_layout = b.layout();
    let a_shape = a_layout.dims();
    let b_shape = b_layout.dims();
    let out_shape = compute_broadcast_shape(a_shape, b_shape);
    let max_rank = out_shape.len();

    // Pad inputs to match output rank
    let a_padded = pad_for_broadcast(a, max_rank);
    let b_padded = pad_for_broadcast(b, max_rank);

    // Create a temporary tensor with output shape to use as reference for repeat
    let out_ref = ctxt.tensor_uninit::<f32>(&out_shape)?;

    // Broadcast a to output shape using repeat
    let mut out = ctxt.repeat(a_padded, &out_ref)?;

    // Apply the operation (sub_assign, mul_assign, div_assign)
    op(ctxt, &mut out, b_padded)?;

    Ok(out)
}

// =============================================================================
// Unary operations
// =============================================================================

fn dispatch_unary<'a>(
    node: &OnnxNode,
    inputs: &HashMap<String, TensorRef<'a, f32>>,
    ctxt: &mut LlmContext<'_>,
    op: UnaryOp,
) -> Result<Vec<CachedTensor<f32>>, OnnxError> {
    let x = get_input(node, inputs, 0)?;
    let out = ctxt.unop(op, x)?;
    Ok(vec![out])
}

fn dispatch_elu<'a>(
    node: &OnnxNode,
    inputs: &HashMap<String, TensorRef<'a, f32>>,
    ctxt: &mut LlmContext<'_>,
) -> Result<Vec<CachedTensor<f32>>, OnnxError> {
    let _alpha = node.get_float_attr_or("alpha", 1.0);
    // TODO: Pass alpha to the kernel if it supports it
    let x = get_input(node, inputs, 0)?;
    let out = ctxt.unop(UnaryOp::Elu, x)?;
    Ok(vec![out])
}

fn dispatch_leaky_relu<'a>(
    node: &OnnxNode,
    inputs: &HashMap<String, TensorRef<'a, f32>>,
    ctxt: &mut LlmContext<'_>,
) -> Result<Vec<CachedTensor<f32>>, OnnxError> {
    let alpha = node.get_float_attr_or("alpha", 0.01);
    let x = get_input(node, inputs, 0)?;
    let out = ctxt.unop_with_args(UnaryOp::LeakyRelu, x, [alpha, 0.0, 0.0, 0.0].into())?;
    Ok(vec![out])
}

fn dispatch_clip<'a>(
    node: &OnnxNode,
    inputs: &HashMap<String, TensorRef<'a, f32>>,
    ctxt: &mut LlmContext<'_>,
) -> Result<Vec<CachedTensor<f32>>, OnnxError> {
    let x = get_input(node, inputs, 0)?;

    // ONNX Clip has min/max as optional inputs (opset >= 11) or attributes (opset < 11)
    let min_val = if let Some(min_tensor) = get_optional_input(node, inputs, 1) {
        // Read scalar from tensor - for now just use -inf if we can't read it
        // TODO: Actually read the scalar value from GPU
        let _ = min_tensor;
        f32::NEG_INFINITY
    } else {
        node.get_float_attr_or("min", f32::NEG_INFINITY)
    };

    let max_val = if let Some(max_tensor) = get_optional_input(node, inputs, 2) {
        let _ = max_tensor;
        f32::INFINITY
    } else {
        node.get_float_attr_or("max", f32::INFINITY)
    };

    let out = ctxt.unop_with_args(UnaryOp::Clamp, x, [min_val, max_val, 0.0, 0.0].into())?;
    Ok(vec![out])
}

// =============================================================================
// Binary operations
// =============================================================================

fn dispatch_binary_add<'a>(
    node: &OnnxNode,
    inputs: &HashMap<String, TensorRef<'a, f32>>,
    ctxt: &mut LlmContext<'_>,
) -> Result<Vec<CachedTensor<f32>>, OnnxError> {
    let a = get_input(node, inputs, 0)?;
    let b = get_input(node, inputs, 1)?;
    let out = dispatch_broadcast_binary_op(a, b, ctxt, |ctxt, out, b| ctxt.add_assign(out, b))?;
    Ok(vec![out])
}

fn dispatch_binary_sub<'a>(
    node: &OnnxNode,
    inputs: &HashMap<String, TensorRef<'a, f32>>,
    ctxt: &mut LlmContext<'_>,
) -> Result<Vec<CachedTensor<f32>>, OnnxError> {
    let a = get_input(node, inputs, 0)?;
    let b = get_input(node, inputs, 1)?;
    let out = dispatch_broadcast_binary_op(a, b, ctxt, |ctxt, out, b| ctxt.sub_assign(out, b))?;
    Ok(vec![out])
}

fn dispatch_binary_mul<'a>(
    node: &OnnxNode,
    inputs: &HashMap<String, TensorRef<'a, f32>>,
    ctxt: &mut LlmContext<'_>,
) -> Result<Vec<CachedTensor<f32>>, OnnxError> {
    let a = get_input(node, inputs, 0)?;
    let b = get_input(node, inputs, 1)?;
    let out = dispatch_broadcast_binary_op(a, b, ctxt, |ctxt, out, b| ctxt.mul_assign(out, b))?;
    Ok(vec![out])
}

fn dispatch_binary_div<'a>(
    node: &OnnxNode,
    inputs: &HashMap<String, TensorRef<'a, f32>>,
    ctxt: &mut LlmContext<'_>,
) -> Result<Vec<CachedTensor<f32>>, OnnxError> {
    let a = get_input(node, inputs, 0)?;
    let b = get_input(node, inputs, 1)?;
    let out = dispatch_broadcast_binary_op(a, b, ctxt, |ctxt, out, b| ctxt.div_assign(out, b))?;
    Ok(vec![out])
}

// =============================================================================
// Matrix operations
// =============================================================================

fn dispatch_matmul<'a>(
    node: &OnnxNode,
    inputs: &HashMap<String, TensorRef<'a, f32>>,
    ctxt: &mut LlmContext<'_>,
) -> Result<Vec<CachedTensor<f32>>, OnnxError> {
    let a = get_input(node, inputs, 0)?;
    let b = get_input(node, inputs, 1)?;

    let a_rank = a.layout().rank as usize;
    let b_rank = b.layout().rank as usize;

    // For batch matmul with broadcasting, we need to handle cases like:
    // [1, 192, 197, 192] x [1, 1, 192, 192] -> need to broadcast b to [1, 192, 192, 192]
    // But for simple cases like [1, 197, 192] x [192, 768] -> just use direct matmul

    // Only apply batch broadcasting when both tensors have batch dimensions (rank >= 3)
    // and those batch dimensions differ
    if a_rank >= 3 && b_rank >= 3 {
        let max_rank = a_rank.max(b_rank);
        let a_padded = pad_for_broadcast(a, max_rank);
        let b_padded = pad_for_broadcast(b, max_rank);

        // Check if batch dimensions need broadcasting (all dims except last 2)
        let a_layout = a_padded.layout();
        let b_layout = b_padded.layout();
        let a_shape: Vec<u32> = a_layout.dims().to_vec();
        let b_shape: Vec<u32> = b_layout.dims().to_vec();

        let mut need_broadcast = false;
        let mut b_target_shape = b_shape.clone();
        for i in 0..(max_rank - 2) {
            if a_shape[i] != b_shape[i] && b_shape[i] == 1 {
                b_target_shape[i] = a_shape[i];
                need_broadcast = true;
            }
        }

        if need_broadcast {
            // Broadcast b to match a's batch dimensions
            let b_ref = ctxt.tensor_uninit::<f32>(&b_target_shape)?;
            let b_broadcast = ctxt.repeat(b_padded, &b_ref)?;
            let out = ctxt.matmul(a_padded, &b_broadcast)?;
            return Ok(vec![out]);
        }

        let out = ctxt.matmul(a_padded, b_padded)?;
        return Ok(vec![out]);
    }

    // For cases where one tensor has batch dims and the other doesn't,
    // just use direct matmul - vortx handles this internally
    let out = ctxt.matmul(a, b)?;
    Ok(vec![out])
}

fn dispatch_gemm<'a>(
    node: &OnnxNode,
    inputs: &HashMap<String, TensorRef<'a, f32>>,
    ctxt: &mut LlmContext<'_>,
) -> Result<Vec<CachedTensor<f32>>, OnnxError> {
    let a = get_input(node, inputs, 0)?;
    let b = get_input(node, inputs, 1)?;
    let c = get_optional_input(node, inputs, 2);

    let alpha = node.get_float_attr_or("alpha", 1.0);
    let beta = node.get_float_attr_or("beta", 1.0);
    let trans_a = node.get_int_attr_or("transA", 0) != 0;
    let trans_b = node.get_int_attr_or("transB", 0) != 0;

    // Apply transpositions
    let a = if trans_a { a.transpose_last_dims() } else { a };
    let b = if trans_b { b.transpose_last_dims() } else { b };

    // Compute A @ B
    let mut out = ctxt.matmul(a, b)?;

    // Apply alpha scaling if needed
    if (alpha - 1.0).abs() > 1e-6 {
        ctxt.scale_assign(&mut out, alpha)?;
    }

    // Add bias C if present
    if let Some(c) = c {
        if (beta - 1.0).abs() > 1e-6 {
            // out += beta * c
            let scaled_c = ctxt.scale(c, beta)?;
            ctxt.add_assign(&mut out, &scaled_c)?;
        } else {
            ctxt.add_assign(&mut out, c)?;
        }
    }

    Ok(vec![out])
}

// =============================================================================
// Normalization operations
// =============================================================================

fn dispatch_softmax<'a>(
    node: &OnnxNode,
    inputs: &HashMap<String, TensorRef<'a, f32>>,
    ctxt: &mut LlmContext<'_>,
) -> Result<Vec<CachedTensor<f32>>, OnnxError> {
    let x = get_input(node, inputs, 0)?;
    // ONNX Softmax axis defaults to 1 for opset versions < 13, and -1 for opset >= 13
    // We use 1 as default since many older models expect this
    let axis = node.get_int_attr_or("axis", 1);

    // Normalize axis
    let rank = x.layout().rank as i64;
    let axis = if axis < 0 { rank + axis } else { axis } as usize;

    // Make a copy since softmax is in-place
    let mut out = ctxt.copy_tensor(x)?;

    let x_layout = x.layout();
    let x_shape = x_layout.dims();

    // Our softmax operates on rows (last dimension) or cols (second-to-last)
    // For other axes, we handle special cases
    if axis == (rank as usize - 1) {
        ctxt.softmax_rows(&mut out)?;
    } else if axis == (rank as usize - 2) {
        ctxt.softmax_cols(&mut out)?;
    } else if axis == 1 && rank == 4 {
        // Special case: softmax over channels in NCHW format
        // For shape [N, C, H, W], we need to transpose to [N, H, W, C], apply softmax_rows, transpose back
        // However, if H=1 and W=1, we can simply squeeze/unsqueeze
        if x_shape[2] == 1 && x_shape[3] == 1 {
            // [N, C, 1, 1] -> apply softmax on C dimension
            // We can reshape to [N, C] and use softmax_rows since C is now the last dim
            let n = x_shape[0];
            let c = x_shape[1];
            let mut reshaped = out.tensor_mut().reshape_mut(&[n, c]);
            ctxt.softmax_rows(&mut reshaped)?;
            // Output shape is already [N, C, 1, 1]
        } else {
            // General case: need to transpose and reshape
            return Err(OnnxError::UnsupportedOp {
                op: format!("Softmax with axis={} on shape {:?}", axis, x_shape),
                node: node.name.clone(),
            });
        }
    } else {
        return Err(OnnxError::UnsupportedOp {
            op: format!("Softmax with axis={}", axis),
            node: node.name.clone(),
        });
    }

    Ok(vec![out])
}

fn dispatch_layernorm<'a>(
    node: &OnnxNode,
    inputs: &HashMap<String, TensorRef<'a, f32>>,
    ctxt: &mut LlmContext<'_>,
) -> Result<Vec<CachedTensor<f32>>, OnnxError> {
    let x = get_input(node, inputs, 0)?;
    let _scale = get_optional_input(node, inputs, 1);
    let _bias = get_optional_input(node, inputs, 2);
    let eps = node.get_float_attr_or("epsilon", 1e-5);

    // TODO: Handle scale and bias properly
    // For now, just do basic layer normalization
    let out = ctxt.layernorm(x, eps)?;

    Ok(vec![out])
}

// =============================================================================
// Shape query operation
// =============================================================================

/// Shape operation - returns the shape of the input tensor as a 1D tensor.
fn dispatch_shape<'a>(
    node: &OnnxNode,
    inputs: &HashMap<String, TensorRef<'a, f32>>,
    ctxt: &mut LlmContext<'_>,
) -> Result<Vec<CachedTensor<f32>>, OnnxError> {
    use khal::BufferUsages;
    use vortx::tensor::TensorBuilder;

    let x = get_input(node, inputs, 0)?;
    let x_layout = x.layout();
    let x_shape = x_layout.dims();

    // Get optional start/end attributes (ONNX opset 15+)
    let start = node.get_int_attr_or("start", 0);
    let end = node.get_int_attr_or("end", x_shape.len() as i64);

    // Normalize negative indices
    let rank = x_shape.len() as i64;
    let start = if start < 0 {
        (rank + start).max(0)
    } else {
        start.min(rank)
    } as usize;
    let end = if end < 0 {
        (rank + end).max(0)
    } else {
        end.min(rank)
    } as usize;

    // Extract the shape slice
    let shape_slice = &x_shape[start..end];

    // Create a tensor containing the shape values (as f32 since our system uses f32)
    let shape_data: Vec<f32> = shape_slice.iter().map(|&d| d as f32).collect();
    let usage = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;
    let tensor = TensorBuilder::tensor(&[shape_data.len() as u32], usage)
        .build_init(ctxt.backend, &shape_data)
        .map_err(OnnxError::GpuError)?;
    let out = ctxt.cache.enroll(tensor, usage);

    Ok(vec![out])
}

// =============================================================================
// Shape manipulation operations
// =============================================================================

fn dispatch_reshape<'a>(
    node: &OnnxNode,
    inputs: &HashMap<String, TensorRef<'a, f32>>,
    output_shapes: &[Vec<u32>],
    ctxt: &mut LlmContext<'_>,
) -> Result<Vec<CachedTensor<f32>>, OnnxError> {
    let x = get_input(node, inputs, 0)?;

    // Use the pre-computed output shape from shape inference
    let target_shape = output_shapes
        .first()
        .ok_or_else(|| OnnxError::ShapeInferenceError {
            node: node.name.clone(),
            reason: "No output shape available for Reshape".to_string(),
        })?;

    // Copy the input data into a tensor with the target shape
    let out = ctxt.copy_tensor_with_shape(x, target_shape)?;
    Ok(vec![out])
}

fn dispatch_transpose<'a>(
    node: &OnnxNode,
    inputs: &HashMap<String, TensorRef<'a, f32>>,
    ctxt: &mut LlmContext<'_>,
) -> Result<Vec<CachedTensor<f32>>, OnnxError> {
    let x = get_input(node, inputs, 0)?;
    let rank = x.layout().rank as usize;

    // Get permutation, default is reverse of dimensions
    let perm = node.get_ints_attr_or("perm", &[]);

    let transposed = if perm.is_empty() {
        // Default: reverse all dimensions
        // For 2D, this is just transpose_last_dims
        if rank == 2 {
            x.transpose_last_dims()
        } else {
            // Build reverse permutation: [rank-1, rank-2, ..., 0, rank, rank+1, ...]
            // ONNX convention: new_size[k] = old_size[perm[k]], so reverse is perm[k] = rank - 1 - k
            // vortx convention: new_size[perm[k]] = old_size[k], so we need the inverse
            // For reverse permutation, the inverse is also reverse
            let mut perm_arr = [0usize, 1, 2, 3];
            for i in 0..rank {
                perm_arr[i] = rank - 1 - i;
            }
            x.permute(perm_arr)
        }
    } else if perm.len() != rank {
        return Err(OnnxError::UnsupportedOp {
            op: format!("Transpose with perm={:?} but rank={}", perm, rank),
            node: node.name.clone(),
        });
    } else {
        // ONNX convention: new_size[k] = old_size[perm[k]]
        // vortx convention: new_size[perm[k]] = old_size[k]
        // So we need to compute the inverse permutation for vortx
        let mut vortx_perm = [0usize, 1, 2, 3];
        for i in 0..rank {
            vortx_perm[perm[i] as usize] = i;
        }
        x.permute(vortx_perm)
    };

    // Make contiguous copy of the transposed view
    let out = ctxt.copy_tensor(transposed)?;
    Ok(vec![out])
}

fn dispatch_squeeze<'a>(
    node: &OnnxNode,
    inputs: &HashMap<String, TensorRef<'a, f32>>,
    output_shapes: &[Vec<u32>],
    ctxt: &mut LlmContext<'_>,
) -> Result<Vec<CachedTensor<f32>>, OnnxError> {
    let x = get_input(node, inputs, 0)?;

    // Use the pre-computed output shape
    let target_shape = output_shapes
        .first()
        .ok_or_else(|| OnnxError::ShapeInferenceError {
            node: node.name.clone(),
            reason: "No output shape available for Squeeze".to_string(),
        })?;

    let out = ctxt.copy_tensor_with_shape(x, target_shape)?;
    Ok(vec![out])
}

fn dispatch_unsqueeze<'a>(
    node: &OnnxNode,
    inputs: &HashMap<String, TensorRef<'a, f32>>,
    output_shapes: &[Vec<u32>],
    ctxt: &mut LlmContext<'_>,
) -> Result<Vec<CachedTensor<f32>>, OnnxError> {
    let x = get_input(node, inputs, 0)?;

    // Use the pre-computed output shape
    let target_shape = output_shapes
        .first()
        .ok_or_else(|| OnnxError::ShapeInferenceError {
            node: node.name.clone(),
            reason: "No output shape available for Unsqueeze".to_string(),
        })?;

    let out = ctxt.copy_tensor_with_shape(x, target_shape)?;
    Ok(vec![out])
}

fn dispatch_flatten<'a>(
    node: &OnnxNode,
    inputs: &HashMap<String, TensorRef<'a, f32>>,
    output_shapes: &[Vec<u32>],
    ctxt: &mut LlmContext<'_>,
) -> Result<Vec<CachedTensor<f32>>, OnnxError> {
    let x = get_input(node, inputs, 0)?;

    // Use the pre-computed output shape
    let target_shape = output_shapes
        .first()
        .ok_or_else(|| OnnxError::ShapeInferenceError {
            node: node.name.clone(),
            reason: "No output shape available for Flatten".to_string(),
        })?;

    let out = ctxt.copy_tensor_with_shape(x, target_shape)?;
    Ok(vec![out])
}

// =============================================================================
// Identity / Copy
// =============================================================================

fn dispatch_identity<'a>(
    node: &OnnxNode,
    inputs: &HashMap<String, TensorRef<'a, f32>>,
    ctxt: &mut LlmContext<'_>,
) -> Result<Vec<CachedTensor<f32>>, OnnxError> {
    let x = get_input(node, inputs, 0)?;
    let out = ctxt.copy_tensor(x)?;
    Ok(vec![out])
}

// =============================================================================
// Constant
// =============================================================================

fn dispatch_constant(
    node: &OnnxNode,
    ctxt: &mut LlmContext<'_>,
) -> Result<Vec<CachedTensor<f32>>, OnnxError> {
    use crate::onnx::tensor::{tensor_proto_shape, tensor_proto_to_f32_data};
    use khal::BufferUsages;
    use vortx::tensor::TensorBuilder;

    let usage = BufferUsages::STORAGE | BufferUsages::COPY_DST;

    // Get constant value from attribute
    if let Ok(tensor_proto) = node.get_tensor_attr("value") {
        let data = tensor_proto_to_f32_data(tensor_proto)?;
        let shape = tensor_proto_shape(tensor_proto);
        let tensor = TensorBuilder::tensor(&shape, usage)
            .build_init(ctxt.backend, &data)
            .map_err(OnnxError::GpuError)?;
        let out = ctxt.cache.enroll(tensor, usage);
        return Ok(vec![out]);
    }

    // Handle scalar constants
    if let Ok(value) = node.get_float_attr("value_float") {
        let tensor = TensorBuilder::tensor(&[], usage)
            .build_init(ctxt.backend, &[value])
            .map_err(OnnxError::GpuError)?;
        let out = ctxt.cache.enroll(tensor, usage);
        return Ok(vec![out]);
    }

    if let Ok(value) = node.get_int_attr("value_int") {
        let tensor = TensorBuilder::tensor(&[], usage)
            .build_init(ctxt.backend, &[value as f32])
            .map_err(OnnxError::GpuError)?;
        let out = ctxt.cache.enroll(tensor, usage);
        return Ok(vec![out]);
    }

    Err(OnnxError::InvalidAttribute {
        attr: "value".to_string(),
        node: node.name.clone(),
        reason: "Constant node missing value attribute".to_string(),
    })
}

// =============================================================================
// Pow operation
// =============================================================================

fn dispatch_pow<'a>(
    node: &OnnxNode,
    inputs: &HashMap<String, TensorRef<'a, f32>>,
    ctxt: &mut LlmContext<'_>,
) -> Result<Vec<CachedTensor<f32>>, OnnxError> {
    let x = get_input(node, inputs, 0)?;
    let y = get_input(node, inputs, 1)?;

    // For now, assume y is a scalar exponent
    // TODO: Support element-wise pow
    // We'll read the first element as the exponent
    let y_len = y.len();
    if y_len == 1 {
        // Scalar exponent - we need to get the value
        // For now, use a default value. In practice, we'd need to read from GPU
        // TODO: Read the actual scalar value from the y tensor
        let exponent = 2.0f32; // Placeholder - should read from y
        let out = ctxt.unop_with_args(UnaryOp::Pow, x, [exponent, 0.0, 0.0, 0.0].into())?;
        Ok(vec![out])
    } else {
        Err(OnnxError::UnsupportedOp {
            op: "Pow with non-scalar exponent".to_string(),
            node: node.name.clone(),
        })
    }
}

// =============================================================================
// Gather operation
// =============================================================================

fn dispatch_gather<'a>(
    node: &OnnxNode,
    inputs: &HashMap<String, TensorRef<'a, f32>>,
    ctxt: &mut LlmContext<'_>,
) -> Result<Vec<CachedTensor<f32>>, OnnxError> {
    use khal::BufferUsages;
    use vortx::tensor::TensorBuilder;

    let data = get_input(node, inputs, 0)?;
    let _indices_f32 = get_input(node, inputs, 1)?;

    let axis = node.get_int_attr_or("axis", 0);
    let axis = if axis < 0 {
        (data.layout().rank as i64 + axis) as u32
    } else {
        axis as u32
    };

    // Get the indices tensor shape to determine output shape
    let indices_layout = _indices_f32.layout();
    let indices_shape = indices_layout.dims();
    let data_layout = data.layout();
    let data_shape = data_layout.dims();

    // Compute output shape: replace the axis dimension with indices shape
    let mut output_shape: Vec<u32> = Vec::new();
    for (i, &dim) in data_shape.iter().enumerate() {
        if i == axis as usize {
            output_shape.extend_from_slice(indices_shape);
        } else {
            output_shape.push(dim);
        }
    }

    // For now, we need to create an i32 indices tensor from the f32 input
    // This is a workaround since ONNX stores indices as int64 but our kernel uses i32
    // In practice, the indices would already be i32 on the GPU
    // TODO: Support reading indices properly

    // Create a placeholder indices tensor
    let indices_len = _indices_f32.len() as usize;
    let indices_data: Vec<i32> = (0..indices_len).map(|i| i as i32).collect();
    let indices = TensorBuilder::tensor(indices_shape, BufferUsages::STORAGE)
        .build_init(ctxt.backend, &indices_data)
        .map_err(OnnxError::GpuError)?;

    let out = ctxt.gather(data, &indices, axis, &output_shape)?;
    Ok(vec![out])
}

// =============================================================================
// Concat operation
// =============================================================================

fn dispatch_concat<'a>(
    node: &OnnxNode,
    inputs: &HashMap<String, TensorRef<'a, f32>>,
    ctxt: &mut LlmContext<'_>,
) -> Result<Vec<CachedTensor<f32>>, OnnxError> {
    let axis = node.get_int_attr_or("axis", 0);

    // Collect all input tensors
    let mut input_tensors: Vec<TensorRef<'a, f32>> = Vec::new();
    for input_name in &node.inputs {
        if input_name.is_empty() {
            continue;
        }
        if let Some(tensor) = inputs.get(input_name) {
            input_tensors.push(*tensor);
        }
    }

    if input_tensors.is_empty() {
        return Err(OnnxError::MissingInput(format!(
            "Concat node '{}' has no inputs",
            node.name
        )));
    }

    // Get the first tensor's shape to determine output shape
    let first_layout = input_tensors[0].layout();
    let first_shape = first_layout.dims();
    let rank = first_shape.len();

    // Normalize axis
    let axis = if axis < 0 {
        (rank as i64 + axis) as usize
    } else {
        axis as usize
    };

    // Compute output shape
    let mut output_shape: Vec<u32> = first_shape.to_vec();
    let mut total_size_along_axis = first_shape[axis];
    for tensor in input_tensors.iter().skip(1) {
        total_size_along_axis += tensor.layout().dims()[axis];
    }
    output_shape[axis] = total_size_along_axis;

    // Allocate output tensor
    let mut output = ctxt.tensor_uninit(&output_shape)?;

    // Copy each input into the output at the appropriate offset
    let mut offset: u32 = 0;
    for tensor in &input_tensors {
        let size_along_axis = tensor.layout().dims()[axis];
        ctxt.concat_copy(&mut output, *tensor, axis as u32, offset)?;
        offset += size_along_axis;
    }

    Ok(vec![output])
}

// =============================================================================
// Reduce operations
// =============================================================================

fn dispatch_reduce_sum<'a>(
    node: &OnnxNode,
    inputs: &HashMap<String, TensorRef<'a, f32>>,
    ctxt: &mut LlmContext<'_>,
) -> Result<Vec<CachedTensor<f32>>, OnnxError> {
    dispatch_reduce_op(node, inputs, ctxt, "sum")
}

fn dispatch_reduce_mean<'a>(
    node: &OnnxNode,
    inputs: &HashMap<String, TensorRef<'a, f32>>,
    ctxt: &mut LlmContext<'_>,
) -> Result<Vec<CachedTensor<f32>>, OnnxError> {
    dispatch_reduce_op(node, inputs, ctxt, "mean")
}

fn dispatch_reduce_max<'a>(
    node: &OnnxNode,
    inputs: &HashMap<String, TensorRef<'a, f32>>,
    ctxt: &mut LlmContext<'_>,
) -> Result<Vec<CachedTensor<f32>>, OnnxError> {
    dispatch_reduce_op(node, inputs, ctxt, "max")
}

fn dispatch_reduce_min<'a>(
    node: &OnnxNode,
    inputs: &HashMap<String, TensorRef<'a, f32>>,
    ctxt: &mut LlmContext<'_>,
) -> Result<Vec<CachedTensor<f32>>, OnnxError> {
    dispatch_reduce_op(node, inputs, ctxt, "min")
}

fn dispatch_reduce_op<'a>(
    node: &OnnxNode,
    inputs: &HashMap<String, TensorRef<'a, f32>>,
    ctxt: &mut LlmContext<'_>,
    op_name: &str,
) -> Result<Vec<CachedTensor<f32>>, OnnxError> {
    let data = get_input(node, inputs, 0)?;
    let data_layout = data.layout();
    let data_shape = data_layout.dims();
    let rank = data_shape.len();

    // Get axes to reduce
    let axes = node.get_ints_attr_or("axes", &[]);
    let keepdims = node.get_int_attr_or("keepdims", 1) != 0;

    // If no axes specified, reduce all dimensions
    let axes: Vec<i64> = if axes.is_empty() {
        (0..rank as i64).collect()
    } else {
        axes.to_vec()
    };

    // Normalize negative axes
    let mut normalized_axes: Vec<usize> = axes
        .iter()
        .map(|&a| {
            if a < 0 {
                (rank as i64 + a) as usize
            } else {
                a as usize
            }
        })
        .collect();
    normalized_axes.sort();
    normalized_axes.dedup();

    // For now, we only support reducing a single axis at a time
    // We'll iterate and reduce one axis at a time
    let mut current = ctxt.copy_tensor(data)?;

    for (reduction_idx, &axis) in normalized_axes.iter().enumerate() {
        let current_shape = current.shape();
        let current_rank = current.rank() as usize;

        // Adjust axis for previous reductions if keepdims is false
        let adjusted_axis = if keepdims { axis } else { axis - reduction_idx };

        if adjusted_axis >= current_rank {
            continue;
        }

        // Compute output shape for this reduction
        let mut output_shape: Vec<u32> = Vec::new();
        for (i, &dim) in current_shape.iter().take(current_rank).enumerate() {
            if i == adjusted_axis {
                if keepdims {
                    output_shape.push(1);
                }
            } else {
                output_shape.push(dim);
            }
        }

        if output_shape.is_empty() {
            output_shape.push(1); // Scalar output
        }

        // Perform reduction
        let reduced = match op_name {
            "sum" => ctxt.reduce_sum_axis(&current, adjusted_axis as u32, &output_shape)?,
            "mean" => ctxt.reduce_mean_axis(&current, adjusted_axis as u32, &output_shape)?,
            "max" => ctxt.reduce_max_axis(&current, adjusted_axis as u32, &output_shape)?,
            "min" => ctxt.reduce_min_axis(&current, adjusted_axis as u32, &output_shape)?,
            _ => {
                return Err(OnnxError::UnsupportedOp {
                    op: format!("Reduce{}", op_name),
                    node: node.name.clone(),
                })
            }
        };

        current = reduced;
    }

    Ok(vec![current])
}

// =============================================================================
// Convolution operation
// =============================================================================

fn dispatch_conv<'a>(
    node: &OnnxNode,
    inputs: &HashMap<String, TensorRef<'a, f32>>,
    ctxt: &mut LlmContext<'_>,
) -> Result<Vec<CachedTensor<f32>>, OnnxError> {
    let x = get_input(node, inputs, 0)?;
    let w = get_input(node, inputs, 1)?;
    let bias = get_optional_input(node, inputs, 2);

    // Get attributes
    let strides = node.get_ints_attr_or("strides", &[1, 1]);
    let pads = node.get_ints_attr_or("pads", &[0, 0, 0, 0]);
    let dilations = node.get_ints_attr_or("dilations", &[1, 1]);
    let groups = node.get_int_attr_or("group", 1) as u32;
    let auto_pad = node.get_string_attr_or("auto_pad", "NOTSET");

    // Get kernel shape from weights
    let w_layout = w.layout();
    let kh = w_layout.size[2];
    let kw = w_layout.size[3];

    // ONNX uses [stride_h, stride_w] format
    let stride_h = strides.first().copied().unwrap_or(1) as u32;
    let stride_w = strides.get(1).copied().unwrap_or(stride_h as i64) as u32;
    let dilation_h = dilations.first().copied().unwrap_or(1) as u32;
    let dilation_w = dilations.get(1).copied().unwrap_or(1) as u32;

    // Compute padding - handle auto_pad
    let (pad_h, pad_w) = if auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER" {
        // For SAME padding, compute needed padding
        let x_layout = x.layout();
        let ih = x_layout.size[2];
        let iw = x_layout.size[3];
        // Output size = ceil(input / stride)
        let oh = ih.div_ceil(stride_h);
        let ow = iw.div_ceil(stride_w);
        // Total padding = max(0, (output - 1) * stride + dilation * (kernel - 1) + 1 - input)
        let total_pad_h = ((oh - 1) * stride_h + dilation_h * (kh - 1) + 1).saturating_sub(ih);
        let total_pad_w = ((ow - 1) * stride_w + dilation_w * (kw - 1) + 1).saturating_sub(iw);
        // Split padding - SAME_UPPER puts extra on bottom/right, SAME_LOWER on top/left
        // For simplicity, we use symmetric padding (half on each side)
        (total_pad_h / 2, total_pad_w / 2)
    } else {
        // VALID or explicit padding
        let pad_h = pads.first().copied().unwrap_or(0) as u32;
        let pad_w = pads.get(1).copied().unwrap_or(0) as u32;
        (pad_h, pad_w)
    };

    // Use NCHW-compatible convolution with groups support
    let mut out = ctxt.conv_2d_nchw(
        x, w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups,
    )?;

    // Add bias if present
    if let Some(bias_tensor) = bias {
        // Bias shape is [OC], need to broadcast it to [N, OC, H, W]
        // Use add_bias_nchw which handles the reshape from [C] to [1, C, 1, 1]
        out = ctxt.add_bias_nchw(&out, bias_tensor)?;
    }

    Ok(vec![out])
}

// =============================================================================
// Pooling operations
// =============================================================================

fn dispatch_max_pool<'a>(
    node: &OnnxNode,
    inputs: &HashMap<String, TensorRef<'a, f32>>,
    ctxt: &mut LlmContext<'_>,
) -> Result<Vec<CachedTensor<f32>>, OnnxError> {
    let x = get_input(node, inputs, 0)?;

    // Get attributes
    let kernel_shape = node.get_ints_attr_or("kernel_shape", &[]);
    let strides = node.get_ints_attr_or("strides", &[1, 1]);
    let pads = node.get_ints_attr_or("pads", &[0, 0, 0, 0]);

    if kernel_shape.len() < 2 {
        return Err(OnnxError::InvalidAttribute {
            attr: "kernel_shape".to_string(),
            node: node.name.clone(),
            reason: "MaxPool requires kernel_shape with at least 2 dimensions".to_string(),
        });
    }

    let kernel_h = kernel_shape[0] as u32;
    let kernel_w = kernel_shape[1] as u32;
    let stride_h = strides.first().copied().unwrap_or(1) as u32;
    let stride_w = strides.get(1).copied().unwrap_or(1) as u32;
    // ONNX pads format: [pad_h_begin, pad_w_begin, pad_h_end, pad_w_end]
    // For symmetric padding, use (begin + end) / 2
    let ph_begin = pads.first().copied().unwrap_or(0) as u32;
    let pw_begin = pads.get(1).copied().unwrap_or(0) as u32;
    let ph_end = pads.get(2).copied().unwrap_or(0) as u32;
    let pw_end = pads.get(3).copied().unwrap_or(0) as u32;
    let pad_h = (ph_begin + ph_end) / 2;
    let pad_w = (pw_begin + pw_end) / 2;

    let out = ctxt.max_pool_2d(x, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w)?;
    Ok(vec![out])
}

fn dispatch_avg_pool<'a>(
    node: &OnnxNode,
    inputs: &HashMap<String, TensorRef<'a, f32>>,
    ctxt: &mut LlmContext<'_>,
) -> Result<Vec<CachedTensor<f32>>, OnnxError> {
    let x = get_input(node, inputs, 0)?;

    // Get attributes
    let kernel_shape = node.get_ints_attr_or("kernel_shape", &[]);
    let strides = node.get_ints_attr_or("strides", &[1, 1]);
    let pads = node.get_ints_attr_or("pads", &[0, 0, 0, 0]);
    let count_include_pad = node.get_int_attr_or("count_include_pad", 0) != 0;

    if kernel_shape.len() < 2 {
        return Err(OnnxError::InvalidAttribute {
            attr: "kernel_shape".to_string(),
            node: node.name.clone(),
            reason: "AveragePool requires kernel_shape with at least 2 dimensions".to_string(),
        });
    }

    let kernel_h = kernel_shape[0] as u32;
    let kernel_w = kernel_shape[1] as u32;
    let stride_h = strides.first().copied().unwrap_or(1) as u32;
    let stride_w = strides.get(1).copied().unwrap_or(1) as u32;
    // ONNX pads format: [pad_h_begin, pad_w_begin, pad_h_end, pad_w_end]
    let ph_begin = pads.first().copied().unwrap_or(0) as u32;
    let pw_begin = pads.get(1).copied().unwrap_or(0) as u32;
    let ph_end = pads.get(2).copied().unwrap_or(0) as u32;
    let pw_end = pads.get(3).copied().unwrap_or(0) as u32;
    let pad_h = (ph_begin + ph_end) / 2;
    let pad_w = (pw_begin + pw_end) / 2;

    let out = ctxt.avg_pool_2d(
        x,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        count_include_pad,
    )?;
    Ok(vec![out])
}

fn dispatch_global_avg_pool<'a>(
    node: &OnnxNode,
    inputs: &HashMap<String, TensorRef<'a, f32>>,
    ctxt: &mut LlmContext<'_>,
) -> Result<Vec<CachedTensor<f32>>, OnnxError> {
    let x = get_input(node, inputs, 0)?;
    let out = ctxt.global_avg_pool_2d(x)?;
    Ok(vec![out])
}

fn dispatch_global_max_pool<'a>(
    node: &OnnxNode,
    inputs: &HashMap<String, TensorRef<'a, f32>>,
    ctxt: &mut LlmContext<'_>,
) -> Result<Vec<CachedTensor<f32>>, OnnxError> {
    let x = get_input(node, inputs, 0)?;
    let out = ctxt.global_max_pool_2d(x)?;
    Ok(vec![out])
}

// =============================================================================
// BatchNormalization (inference mode)
// =============================================================================

/// BatchNormalization for inference:
/// y = (x - mean) / sqrt(var + epsilon) * scale + bias
///
/// Which can be rewritten as:
/// y = x * k + b
/// where k = scale / sqrt(var + epsilon)
///       b = bias - mean * k
fn dispatch_batch_norm<'a>(
    node: &OnnxNode,
    inputs: &HashMap<String, TensorRef<'a, f32>>,
    ctxt: &mut LlmContext<'_>,
) -> Result<Vec<CachedTensor<f32>>, OnnxError> {
    use crate::ops::UnaryOp;
    use glamx::Vec4;

    // Inputs: X, scale, bias, mean, var
    let x = get_input(node, inputs, 0)?;
    let scale = get_input(node, inputs, 1)?;
    let bias = get_input(node, inputs, 2)?;
    let mean = get_input(node, inputs, 3)?;
    let var = get_input(node, inputs, 4)?;

    let epsilon = node.get_float_attr_or("epsilon", 1e-5);

    // Get dimensions
    let x_layout = x.layout();
    let x_shape = x_layout.dims();
    let channels = x_shape[1];

    // Compute k = scale / sqrt(var + epsilon)
    // First: var + epsilon using AddScalar unary op
    let var_eps =
        ctxt.unop_with_args(UnaryOp::AddScalar, var, Vec4::new(epsilon, 0.0, 0.0, 0.0))?;
    // sqrt(var + epsilon)
    let sqrt_var_eps = ctxt.unop(UnaryOp::Sqrt, &var_eps)?;
    // k = scale / sqrt(var + epsilon)
    let k = ctxt.div(scale, &sqrt_var_eps)?;

    // Compute b = bias - mean * k
    // mean * k
    let mean_k = ctxt.mul(mean, &k)?;
    // b = bias - mean * k
    let b = ctxt.sub(bias, &mean_k)?;

    // Reshape k and b to [1, C, 1, 1] for broadcasting
    let k_view = k.as_tensor_ref().reshape(&[1, channels, 1, 1]);
    let b_view = b.as_tensor_ref().reshape(&[1, channels, 1, 1]);

    // y = x * k + b
    let mut out = ctxt.tensor_uninit(x_shape)?;
    ctxt.copy(x, &mut out)?;
    ctxt.mul_assign(&mut out, k_view)?;
    ctxt.add_assign(&mut out, b_view)?;

    Ok(vec![out])
}

// =============================================================================
// ConstantOfShape - create tensor filled with constant value
// =============================================================================

fn dispatch_constant_of_shape<'a>(
    node: &OnnxNode,
    _inputs: &HashMap<String, TensorRef<'a, f32>>,
    output_shapes: &[Vec<u32>],
    ctxt: &mut LlmContext<'_>,
) -> Result<Vec<CachedTensor<f32>>, OnnxError> {
    use khal::BufferUsages;
    use vortx::tensor::TensorBuilder;

    // Get the fill value from attribute (default is 0.0)
    let fill_value: f32 = if let Ok(tensor_proto) = node.get_tensor_attr("value") {
        use crate::onnx::tensor::tensor_proto_to_f32_data;
        let data = tensor_proto_to_f32_data(tensor_proto)?;
        data.first().copied().unwrap_or(0.0)
    } else {
        0.0
    };

    // Get output shape from shape inference
    let shape = output_shapes
        .first()
        .ok_or_else(|| OnnxError::ShapeInferenceError {
            node: node.name.clone(),
            reason: "No output shape for ConstantOfShape".to_string(),
        })?;

    // Create tensor filled with the constant value
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let data: Vec<f32> = vec![fill_value; numel];

    let usage = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;
    let tensor = TensorBuilder::tensor(shape, usage)
        .build_init(ctxt.backend, &data)
        .map_err(OnnxError::GpuError)?;

    let out = ctxt.cache.enroll(tensor, usage);
    Ok(vec![out])
}

// =============================================================================
// Slice - extract a slice from tensor
// =============================================================================

fn dispatch_slice<'a>(
    node: &OnnxNode,
    inputs: &HashMap<String, TensorRef<'a, f32>>,
    output_shapes: &[Vec<u32>],
    ctxt: &mut LlmContext<'_>,
) -> Result<Vec<CachedTensor<f32>>, OnnxError> {
    let x = get_input(node, inputs, 0)?;

    // Get output shape from shape inference
    let output_shape = output_shapes
        .first()
        .ok_or_else(|| OnnxError::ShapeInferenceError {
            node: node.name.clone(),
            reason: "No output shape for Slice".to_string(),
        })?;

    // For now, use copy_tensor_with_shape which does a simple reshape
    // A proper slice implementation would need a GPU kernel
    // This is a simplified version that works when slicing doesn't change layout
    let out = ctxt.copy_tensor_with_shape(x, output_shape)?;
    Ok(vec![out])
}

// =============================================================================
// Expand - broadcast tensor to larger shape
// =============================================================================

fn dispatch_expand<'a>(
    node: &OnnxNode,
    inputs: &HashMap<String, TensorRef<'a, f32>>,
    output_shapes: &[Vec<u32>],
    ctxt: &mut LlmContext<'_>,
) -> Result<Vec<CachedTensor<f32>>, OnnxError> {
    let x = get_input(node, inputs, 0)?;

    // Get output shape from shape inference
    let output_shape = output_shapes
        .first()
        .ok_or_else(|| OnnxError::ShapeInferenceError {
            node: node.name.clone(),
            reason: "No output shape for Expand".to_string(),
        })?;

    // Create a tensor with target shape for repeat to use as reference
    let target = ctxt.tensor_uninit(output_shape)?;

    // Use repeat to broadcast x to target's shape
    let out = ctxt.repeat(x, &target)?;
    Ok(vec![out])
}

// =============================================================================
// Where - conditional select
// =============================================================================

fn dispatch_where<'a>(
    node: &OnnxNode,
    inputs: &HashMap<String, TensorRef<'a, f32>>,
    ctxt: &mut LlmContext<'_>,
) -> Result<Vec<CachedTensor<f32>>, OnnxError> {
    // Where(condition, x, y) returns x where condition is true, y otherwise
    // condition is a boolean tensor (stored as f32 where 0 = false, non-zero = true)
    let cond = get_input(node, inputs, 0)?;
    let x = get_input(node, inputs, 1)?;
    let y = get_input(node, inputs, 2)?;

    // For now, simplified implementation: y + cond * (x - y)
    // This works when condition is 0 or 1
    let x_minus_y = ctxt.sub(x, y)?;
    let cond_times_diff = ctxt.mul(cond, &x_minus_y)?;
    let out = ctxt.add(y, &cond_times_diff)?;

    Ok(vec![out])
}

// =============================================================================
// Equal - element-wise equality comparison
// =============================================================================

fn dispatch_equal<'a>(
    node: &OnnxNode,
    inputs: &HashMap<String, TensorRef<'a, f32>>,
    ctxt: &mut LlmContext<'_>,
) -> Result<Vec<CachedTensor<f32>>, OnnxError> {
    // Equal returns 1.0 where a == b, 0.0 otherwise
    // Simplified implementation using: 1 - abs(sign(a - b))
    let a = get_input(node, inputs, 0)?;
    let b = get_input(node, inputs, 1)?;

    // a - b
    let diff = ctxt.sub(a, b)?;
    // abs(a - b)
    let abs_diff = ctxt.unop(crate::ops::UnaryOp::Abs, &diff)?;
    // sign(abs_diff) - returns 0 for 0, 1 for positive
    let sign = ctxt.unop(crate::ops::UnaryOp::Sgn, &abs_diff)?;
    // 1 - sign gives us 1 for equal, 0 for not equal
    let ones = ctxt.scale(&sign, 0.0)?; // Create zeros
    let ones = ctxt.unop_with_args(
        crate::ops::UnaryOp::AddScalar,
        &ones,
        glamx::Vec4::new(1.0, 0.0, 0.0, 0.0),
    )?;
    let out = ctxt.sub(&ones, &sign)?;

    Ok(vec![out])
}
