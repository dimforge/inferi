//! ONNX model loading and compilation.

use crate::context::LlmContext;
use crate::onnx::error::OnnxError;
use crate::onnx::graph::{OnnxGraph, OnnxNode};
use crate::onnx::ops::dispatch_node;
use crate::onnx::tensor::{tensor_proto_shape, tensor_proto_to_f32_data, tensor_proto_to_i64_data};
use crate::tensor_cache::CachedTensor;
use khal::backend::GpuBackend;
use khal::BufferUsages;
use onnx_protobuf::{Message, ModelProto};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use vortx::tensor::{AsTensorRef, Tensor, TensorBuilder, TensorRef};

/// A parsed ONNX model.
pub struct OnnxModel {
    proto: ModelProto,
}

impl OnnxModel {
    /// Load an ONNX model from raw bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, OnnxError> {
        let proto = ModelProto::parse_from_bytes(bytes)?;
        Ok(Self { proto })
    }

    /// Load an ONNX model from a file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, OnnxError> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    /// Get the model's IR version.
    pub fn ir_version(&self) -> i64 {
        self.proto.ir_version
    }

    /// Get the model's opset version.
    pub fn opset_version(&self) -> i64 {
        self.proto
            .opset_import
            .first()
            .map(|o| o.version)
            .unwrap_or(0)
    }

    /// Get the model's producer name.
    pub fn producer_name(&self) -> &str {
        &self.proto.producer_name
    }

    /// Get the model's graph.
    fn graph(&self) -> Result<&onnx_protobuf::GraphProto, OnnxError> {
        self.proto
            .graph
            .0
            .as_ref()
            .map(|b| b.as_ref())
            .ok_or_else(|| OnnxError::ParseError("Model has no graph".to_string()))
    }

    /// Print all operations in the model (for debugging).
    pub fn print_operations(&self) -> Result<(), OnnxError> {
        let graph_proto = self.graph()?;
        let graph = OnnxGraph::from_proto(graph_proto)?;
        println!("Operations in model ({} nodes):", graph.nodes.len());
        for node in &graph.nodes {
            println!("  - {} ({})", node.name, node.op_type);
            // Print relevant attributes for pooling/conv
            if node.op_type.contains("Pool") || node.op_type.contains("Conv") {
                if let Ok(ks) = node.get_ints_attr("kernel_shape") {
                    println!("      kernel_shape: {:?}", ks);
                }
                if let Ok(s) = node.get_ints_attr("strides") {
                    println!("      strides: {:?}", s);
                }
                if let Ok(p) = node.get_ints_attr("pads") {
                    println!("      pads: {:?}", p);
                }
            }
        }
        Ok(())
    }

    /// Get the model's input specifications (names and optional shapes).
    pub fn inputs(&self) -> Result<Vec<crate::onnx::graph::GraphInput>, OnnxError> {
        let graph_proto = self.graph()?;
        let onnx_graph = OnnxGraph::from_proto(graph_proto)?;
        Ok(onnx_graph.inputs)
    }

    /// Get the model's output tensor names.
    pub fn outputs(&self) -> Result<Vec<String>, OnnxError> {
        let graph_proto = self.graph()?;
        let onnx_graph = OnnxGraph::from_proto(graph_proto)?;
        Ok(onnx_graph.outputs)
    }

    /// Compile the model for GPU execution.
    ///
    /// `input_shapes` provides concrete shapes for dynamic dimensions.
    /// Keys are input tensor names, values are the concrete shapes.
    pub fn compile(
        &self,
        backend: &GpuBackend,
        input_shapes: &HashMap<String, Vec<u32>>,
    ) -> Result<CompiledOnnxGraph, OnnxError> {
        let graph_proto = self.graph()?;
        let graph = OnnxGraph::from_proto(graph_proto)?;

        // Upload initializers (model weights) to GPU
        let mut weights: HashMap<String, Tensor<f32>> = HashMap::new();
        for (name, tensor_proto) in &graph.initializers {
            let data = tensor_proto_to_f32_data(tensor_proto)?;
            let shape = tensor_proto_shape(tensor_proto);

            let tensor = TensorBuilder::tensor(
                &shape,
                BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            )
            .build_init(backend, &data)
            .map_err(OnnxError::GpuError)?;

            weights.insert(name.clone(), tensor);
        }

        // Validate and store input shapes
        let mut compiled_input_shapes: HashMap<String, Vec<u32>> = HashMap::new();
        for input in &graph.inputs {
            let shape = input_shapes.get(&input.name).cloned().or_else(|| {
                // Try to get shape from graph definition if it's fully static
                input.shape.as_ref().and_then(|s| {
                    if s.iter().all(|d| d.is_some()) {
                        Some(s.iter().map(|d| d.unwrap()).collect())
                    } else {
                        None
                    }
                })
            });

            if let Some(shape) = shape {
                compiled_input_shapes.insert(input.name.clone(), shape);
            } else {
                return Err(OnnxError::ShapeInferenceError {
                    node: "input".to_string(),
                    reason: format!(
                        "Input '{}' has dynamic dimensions but no shape was provided",
                        input.name
                    ),
                });
            }
        }

        // Infer output shapes for all nodes
        let (tensor_shapes, constant_outputs) =
            infer_shapes(&graph, &compiled_input_shapes, &weights)?;

        Ok(CompiledOnnxGraph {
            graph,
            weights,
            input_shapes: compiled_input_shapes,
            tensor_shapes,
            constant_outputs,
        })
    }
}

/// A compiled ONNX graph ready for execution.
pub struct CompiledOnnxGraph {
    graph: OnnxGraph,
    weights: HashMap<String, Tensor<f32>>,
    input_shapes: HashMap<String, Vec<u32>>,
    tensor_shapes: HashMap<String, Vec<u32>>,
    /// Names of tensors that are constant (computed at compile time for shape inference)
    constant_outputs: HashSet<String>,
}

impl CompiledOnnxGraph {
    /// Get the expected input tensor names and shapes.
    pub fn inputs(&self) -> impl Iterator<Item = (&str, &[u32])> {
        self.input_shapes
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_slice()))
    }

    /// Get the output tensor names.
    pub fn outputs(&self) -> &[String] {
        &self.graph.outputs
    }

    /// Run inference on the compiled graph.
    ///
    /// `inputs` should contain all input tensors required by the model.
    /// Returns a map of output tensor names to their computed values.
    pub async fn run(
        &self,
        ctxt: &mut LlmContext<'_>,
        inputs: HashMap<String, &Tensor<f32>>,
    ) -> Result<HashMap<String, CachedTensor<f32>>, OnnxError> {
        // Validate inputs
        for (name, expected_shape) in &self.input_shapes {
            let tensor = inputs
                .get(name)
                .ok_or_else(|| OnnxError::MissingInput(name.clone()))?;

            let actual_shape: Vec<u32> = tensor.as_tensor_ref().layout().dims().to_vec();
            if actual_shape != *expected_shape {
                return Err(OnnxError::ShapeMismatch {
                    name: name.clone(),
                    expected: expected_shape.clone(),
                    actual: actual_shape,
                });
            }
        }

        // Create tensor map with inputs and weights
        let mut tensors: HashMap<String, TensorHolder> = HashMap::new();

        // Add weights (borrowed)
        for (name, tensor) in &self.weights {
            tensors.insert(name.clone(), TensorHolder::Borrowed(tensor));
        }

        // Add inputs (borrowed)
        for (name, tensor) in &inputs {
            tensors.insert(name.clone(), TensorHolder::Borrowed(tensor));
        }

        // Execute nodes in topological order
        for node in &self.graph.nodes {
            // Skip nodes whose outputs are all constant (computed at compile time for shape inference)
            let all_outputs_constant = node
                .outputs
                .iter()
                .all(|name| name.is_empty() || self.constant_outputs.contains(name));
            if all_outputs_constant {
                continue;
            }

            // Build input refs for this node
            let mut input_refs: HashMap<String, TensorRef<'_, f32>> = HashMap::new();
            for input_name in &node.inputs {
                if input_name.is_empty() {
                    continue;
                }
                if let Some(holder) = tensors.get(input_name) {
                    input_refs.insert(input_name.clone(), holder.as_ref());
                }
            }

            // Get the output shapes for this node
            let output_shapes: Vec<Vec<u32>> = node
                .outputs
                .iter()
                .map(|name| self.tensor_shapes.get(name).cloned().unwrap_or_default())
                .collect();

            // Execute the node
            let outputs = dispatch_node(node, &input_refs, &output_shapes, ctxt)?;

            // Store outputs
            for (output_name, output_tensor) in node.outputs.iter().zip(outputs) {
                if !output_name.is_empty() {
                    tensors.insert(output_name.clone(), TensorHolder::Owned(output_tensor));
                }
            }
        }

        // Collect output tensors
        let mut results: HashMap<String, CachedTensor<f32>> = HashMap::new();
        for output_name in &self.graph.outputs {
            if let Some(holder) = tensors.remove(output_name) {
                match holder {
                    TensorHolder::Owned(tensor) => {
                        results.insert(output_name.clone(), tensor);
                    }
                    TensorHolder::Borrowed(tensor) => {
                        // Need to copy borrowed tensor
                        let copied = ctxt.copy_tensor(tensor)?;
                        results.insert(output_name.clone(), copied);
                    }
                }
            }
        }

        Ok(results)
    }
}

/// Helper enum to hold either owned or borrowed tensors during execution.
enum TensorHolder<'a> {
    Owned(CachedTensor<f32>),
    Borrowed(&'a Tensor<f32>),
}

impl<'a> TensorHolder<'a> {
    fn as_ref(&self) -> TensorRef<'_, f32> {
        match self {
            TensorHolder::Owned(t) => t.as_tensor_ref(),
            TensorHolder::Borrowed(t) => t.as_tensor_ref(),
        }
    }
}

/// Infer tensor shapes for all nodes in the graph.
/// Returns (tensor_shapes, constant_outputs) where constant_outputs contains names
/// of tensors that are computed at compile time (Shape, Slice of Shape, etc.)
#[allow(clippy::type_complexity)]
fn infer_shapes(
    graph: &OnnxGraph,
    input_shapes: &HashMap<String, Vec<u32>>,
    weights: &HashMap<String, Tensor<f32>>,
) -> Result<(HashMap<String, Vec<u32>>, HashSet<String>), OnnxError> {
    let mut shapes: HashMap<String, Vec<u32>> = HashMap::new();
    let mut constant_outputs: HashSet<String> = HashSet::new();

    // Build a map of constant values from initializers (for Reshape shape inputs)
    let mut constant_values: HashMap<String, Vec<i64>> = HashMap::new();
    for (name, proto) in &graph.initializers {
        // Try to extract i64 values (shape tensors are typically int64)
        if let Ok(values) = tensor_proto_to_i64_data(proto) {
            constant_values.insert(name.clone(), values);
        }
    }

    // Add input shapes
    for (name, shape) in input_shapes {
        shapes.insert(name.clone(), shape.clone());
    }

    // Add weight shapes
    for (name, tensor) in weights {
        shapes.insert(
            name.clone(),
            tensor.as_tensor_ref().layout().dims().to_vec(),
        );
    }

    // Infer shapes for each node in topological order
    for node in &graph.nodes {
        let (output_shapes, output_values) = infer_node_shapes(node, &shapes, &constant_values)?;
        for (name, shape) in node.outputs.iter().zip(output_shapes) {
            if !name.is_empty() {
                shapes.insert(name.clone(), shape);
            }
        }
        // Propagate constant values for shape computation
        for (name, values) in node.outputs.iter().zip(output_values) {
            if !name.is_empty() {
                if let Some(v) = values {
                    constant_values.insert(name.clone(), v.clone());
                    // Only mark as "constant output" (skippable at runtime) for shape-related ops
                    // that produce metadata, not actual tensor data. These ops compute shape
                    // information that's only used during graph compilation.
                    let is_shape_metadata_op = matches!(
                        node.op_type.as_str(),
                        "Shape" | "Slice" | "Concat" | "Gather"
                    ) && v.len() <= 8; // Small constant = likely shape metadata
                    if is_shape_metadata_op {
                        constant_outputs.insert(name.clone());
                    }
                }
            }
        }
    }

    Ok((shapes, constant_outputs))
}

/// Infer output shapes for a single node.
/// Returns (output_shapes, output_constant_values) where output_constant_values
/// are optional i64 arrays for outputs that can be computed at compile time (e.g., Shape).
#[allow(clippy::type_complexity)]
fn infer_node_shapes(
    node: &OnnxNode,
    shapes: &HashMap<String, Vec<u32>>,
    constant_values: &HashMap<String, Vec<i64>>,
) -> Result<(Vec<Vec<u32>>, Vec<Option<Vec<i64>>>), OnnxError> {
    let get_shape = |idx: usize| -> Result<Vec<u32>, OnnxError> {
        let name = node
            .inputs
            .get(idx)
            .ok_or_else(|| OnnxError::ShapeInferenceError {
                node: node.name.clone(),
                reason: format!("Missing input at index {}", idx),
            })?;
        if name.is_empty() {
            return Err(OnnxError::ShapeInferenceError {
                node: node.name.clone(),
                reason: format!("Empty input at index {}", idx),
            });
        }
        shapes
            .get(name)
            .cloned()
            .ok_or_else(|| OnnxError::ShapeInferenceError {
                node: node.name.clone(),
                reason: format!("Unknown shape for input '{}'", name),
            })
    };

    let get_constant = |idx: usize| -> Option<Vec<i64>> {
        let name = node.inputs.get(idx)?;
        constant_values.get(name).cloned()
    };

    // Helper to return shapes with no constant values
    let shapes_only = |shapes: Vec<Vec<u32>>| -> (Vec<Vec<u32>>, Vec<Option<Vec<i64>>>) {
        let n = shapes.len();
        (shapes, vec![None; n])
    };

    match node.op_type.as_str() {
        // Unary ops preserve shape
        "Relu" | "Sigmoid" | "Tanh" | "Gelu" | "Silu" | "Abs" | "Neg" | "Sqrt" | "Log" | "Sin"
        | "Cos" | "Elu" | "LeakyRelu" | "HardSigmoid" | "Exp" | "Reciprocal" | "Erf" | "Clip"
        | "Identity" | "Softmax" | "LayerNormalization" | "Dropout" | "BatchNormalization" => {
            Ok(shapes_only(vec![get_shape(0)?]))
        }

        // Pow - output shape matches first input
        "Pow" => Ok(shapes_only(vec![get_shape(0)?])),

        // Binary ops - use broadcasting rules
        "Add" | "Sub" | "Mul" | "Div" => {
            let a_shape = get_shape(0)?;
            let b_shape = get_shape(1)?;
            let out_shape = broadcast_shapes(&a_shape, &b_shape)?;
            Ok(shapes_only(vec![out_shape]))
        }

        // MatMul
        "MatMul" => {
            let a_shape = get_shape(0)?;
            let b_shape = get_shape(1)?;
            let out_shape = matmul_shape(&a_shape, &b_shape)?;
            Ok(shapes_only(vec![out_shape]))
        }

        // Gemm (always 2D output)
        "Gemm" => {
            let mut a_shape = get_shape(0)?;
            let mut b_shape = get_shape(1)?;

            let trans_a = node.get_int_attr_or("transA", 0) != 0;
            let trans_b = node.get_int_attr_or("transB", 0) != 0;

            if trans_a && a_shape.len() == 2 {
                a_shape.swap(0, 1);
            }
            if trans_b && b_shape.len() == 2 {
                b_shape.swap(0, 1);
            }

            // Output is [M, N] where A is [M, K] and B is [K, N]
            let m = *a_shape.first().unwrap_or(&1);
            let n = *b_shape.last().unwrap_or(&1);
            Ok(shapes_only(vec![vec![m, n]]))
        }

        // Constant - shape from attribute, also propagate constant value
        "Constant" => {
            if let Ok(tensor_proto) = node.get_tensor_attr("value") {
                let shape = tensor_proto_shape(tensor_proto);
                // Try to get constant value for shape propagation (for Reshape, etc.)
                let const_val = tensor_proto_to_i64_data(tensor_proto).ok();
                return Ok((vec![shape], vec![const_val]));
            }
            // Scalar constant
            Ok(shapes_only(vec![vec![]]))
        }

        // ConstantOfShape - shape comes from input tensor values
        "ConstantOfShape" => {
            // The input is a 1D tensor containing the shape values
            if let Some(shape_values) = get_constant(0) {
                let shape: Vec<u32> = shape_values.iter().map(|&d| d as u32).collect();
                return Ok(shapes_only(vec![shape]));
            }
            // Fallback to input shape
            Ok(shapes_only(vec![get_shape(0)?]))
        }

        // Equal - output is same shape as input, but boolean
        "Equal" => {
            let a_shape = get_shape(0)?;
            let b_shape = get_shape(1)?;
            let out_shape = broadcast_shapes(&a_shape, &b_shape)?;
            Ok(shapes_only(vec![out_shape]))
        }

        // Expand - broadcast input to given shape
        "Expand" => {
            if let Some(shape_values) = get_constant(1) {
                let shape: Vec<u32> = shape_values.iter().map(|&d| d as u32).collect();
                return Ok(shapes_only(vec![shape]));
            }
            // If shape is not constant, try to infer from input
            let input_shape = get_shape(0)?;
            Ok(shapes_only(vec![input_shape]))
        }

        // Where - output shape is broadcast of condition and inputs
        "Where" => {
            let cond_shape = get_shape(0)?;
            let x_shape = get_shape(1)?;
            let y_shape = get_shape(2)?;
            let mut out_shape = broadcast_shapes(&cond_shape, &x_shape)?;
            out_shape = broadcast_shapes(&out_shape, &y_shape)?;
            Ok(shapes_only(vec![out_shape]))
        }

        // Slice - compute output shape from starts/ends/axes/steps
        "Slice" => {
            let input_shape = get_shape(0)?;
            let starts = get_constant(1).unwrap_or_default();
            let ends = get_constant(2).unwrap_or_default();
            let axes: Vec<i64> =
                get_constant(3).unwrap_or_else(|| (0..input_shape.len() as i64).collect());
            let steps: Vec<i64> = get_constant(4).unwrap_or_else(|| vec![1; axes.len()]);

            // For 1D slicing of constant values (e.g., slicing Shape output)
            if input_shape.len() == 1 && axes.len() == 1 && axes[0] == 0 {
                if let Some(input_values) = get_constant(0) {
                    let dim = input_values.len() as i64;
                    let start = starts.first().copied().unwrap_or(0);
                    let end = ends.first().copied().unwrap_or(dim);
                    let step = steps.first().copied().unwrap_or(1).abs();

                    // Normalize indices
                    let start = if start < 0 {
                        (dim + start).max(0)
                    } else {
                        start.min(dim)
                    } as usize;
                    let end = if end < 0 {
                        (dim + end).max(0)
                    } else {
                        end.min(dim)
                    } as usize;

                    // Extract sliced values
                    let sliced: Vec<i64> = input_values[start..end]
                        .iter()
                        .step_by(step as usize)
                        .copied()
                        .collect();
                    let out_len = sliced.len() as u32;
                    return Ok((vec![vec![out_len]], vec![Some(sliced)]));
                }
            }

            let mut output_shape = input_shape.clone();
            for (i, &axis) in axes.iter().enumerate() {
                let axis = if axis < 0 {
                    input_shape.len() as i64 + axis
                } else {
                    axis
                } as usize;
                if axis >= input_shape.len() {
                    continue;
                }

                let dim = input_shape[axis] as i64;
                let start = if i < starts.len() { starts[i] } else { 0 };
                let end = if i < ends.len() { ends[i] } else { dim };
                let step = if i < steps.len() { steps[i].abs() } else { 1 };

                // Clamp start/end to valid range
                let start = start.clamp(-dim, dim);
                let end = end.clamp(-dim, dim);
                let start = if start < 0 { dim + start } else { start };
                let end = if end < 0 { dim + end } else { end };
                let start = start.max(0).min(dim);
                let end = end.max(0).min(dim);

                let size = ((end - start).abs() + step - 1) / step;
                output_shape[axis] = size.max(0) as u32;
            }
            Ok(shapes_only(vec![output_shape]))
        }

        // Shape operation - returns the shape of a tensor as a 1D tensor
        // Also propagates the shape values as constants
        "Shape" => {
            let input_shape = get_shape(0)?;
            let start = node.get_int_attr_or("start", 0);
            let end = node.get_int_attr_or("end", input_shape.len() as i64);

            // Normalize negative indices
            let rank = input_shape.len() as i64;
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

            // Output is a 1D tensor with size = number of dimensions
            let shape_len = (end - start) as u32;
            // Propagate the actual shape values as constants
            let shape_values: Vec<i64> =
                input_shape[start..end].iter().map(|&d| d as i64).collect();
            Ok((vec![vec![shape_len]], vec![Some(shape_values)]))
        }

        // Gather operation - can propagate constants if input is constant
        "Gather" => {
            let input_shape = get_shape(0)?;
            let indices_shape = get_shape(1)?;
            let axis = node.get_int_attr_or("axis", 0);
            let axis = if axis < 0 {
                input_shape.len() as i64 + axis
            } else {
                axis
            } as usize;

            // Output shape: input_shape[..axis] + indices_shape + input_shape[axis+1..]
            let mut out_shape = Vec::new();
            out_shape.extend_from_slice(&input_shape[..axis]);
            out_shape.extend_from_slice(&indices_shape);
            if axis + 1 < input_shape.len() {
                out_shape.extend_from_slice(&input_shape[axis + 1..]);
            }

            // Try to propagate constant values
            let output_const =
                if let (Some(data), Some(indices)) = (get_constant(0), get_constant(1)) {
                    // Gather elements from data using indices
                    let result: Vec<i64> = indices
                        .iter()
                        .map(|&idx| {
                            let idx = if idx < 0 {
                                data.len() as i64 + idx
                            } else {
                                idx
                            } as usize;
                            data.get(idx).copied().unwrap_or(0)
                        })
                        .collect();
                    Some(result)
                } else {
                    None
                };

            Ok((vec![out_shape], vec![output_const]))
        }

        // Unsqueeze - adds dimensions of size 1
        "Unsqueeze" => {
            let mut shape = get_shape(0)?;
            let axes = if let Some(axes) = get_constant(1) {
                axes
            } else {
                node.get_ints_attr_or("axes", &[]).to_vec()
            };

            // Sort axes in reverse order to insert correctly
            let mut axes: Vec<i64> = axes.to_vec();
            axes.sort_by(|a, b| b.cmp(a));

            for ax in axes {
                let ax = if ax < 0 {
                    shape.len() as i64 + ax + 1
                } else {
                    ax
                } as usize;
                shape.insert(ax, 1);
            }

            // Propagate constant value if input is constant
            let output_const = get_constant(0);
            Ok((vec![shape], vec![output_const]))
        }

        // Shape manipulation ops
        "Reshape" => {
            let input_shape = get_shape(0)?;
            let input_size: u32 = input_shape.iter().product();

            // Try to get shape values from the shape input (second input)
            if let Some(shape_name) = node.inputs.get(1) {
                // eprintln!("Reshape {}: input_shape={:?}, input_size={}, shape_name={}", node.name, input_shape, input_size, shape_name);
                if let Some(shape_values) = constant_values.get(shape_name) {
                    // eprintln!("  shape_values={:?}", shape_values);
                    // Convert i64 shape values to u32, handling -1 (infer dimension)
                    let mut output_shape: Vec<u32> = Vec::new();
                    let mut infer_idx: Option<usize> = None;
                    let mut known_size: u32 = 1;

                    for (i, &dim) in shape_values.iter().enumerate() {
                        if dim == -1 {
                            infer_idx = Some(i);
                            output_shape.push(0); // Placeholder
                        } else if dim == 0 {
                            // 0 means copy from input shape
                            let d = input_shape.get(i).copied().unwrap_or(1);
                            output_shape.push(d);
                            known_size *= d;
                        } else {
                            output_shape.push(dim as u32);
                            known_size *= dim as u32;
                        }
                    }

                    // Infer the -1 dimension
                    if let Some(idx) = infer_idx {
                        if let Some(v) = input_size.checked_div(known_size) {
                            output_shape[idx] = v;
                        }
                    }

                    // Verify output size matches input size
                    let output_size: u32 = output_shape.iter().product();
                    if output_size != input_size && !output_shape.is_empty() {
                        eprintln!("WARNING: Reshape '{}' has mismatched sizes (input={}, output={}). Adjusting last dimension.",
                            node.name, input_size, output_size);
                        // Try to fix by adjusting the last dimension
                        let other_dims: u32 =
                            output_shape.iter().take(output_shape.len() - 1).product();
                        if let Some(v) = input_size.checked_div(other_dims) {
                            let last_idx = output_shape.len() - 1;
                            output_shape[last_idx] = v;
                        }
                    }

                    return Ok(shapes_only(vec![output_shape]));
                }
            }

            // Fallback: return input shape
            Ok(shapes_only(vec![input_shape]))
        }

        "Flatten" => {
            let input_shape = get_shape(0)?;
            let axis = node.get_int_attr_or("axis", 1) as usize;

            // Flatten splits into two dimensions at axis
            let dim0: u32 = input_shape[..axis].iter().product();
            let dim1: u32 = input_shape[axis..].iter().product();

            Ok(shapes_only(vec![vec![dim0.max(1), dim1.max(1)]]))
        }

        "Transpose" => {
            let mut shape = get_shape(0)?;
            let perm = node.get_ints_attr_or("perm", &[]);
            if perm.is_empty() {
                shape.reverse();
            } else if perm.len() == shape.len() {
                let old_shape = shape.clone();
                for (i, &p) in perm.iter().enumerate() {
                    shape[i] = old_shape[p as usize];
                }
            }
            Ok(shapes_only(vec![shape]))
        }

        "Squeeze" => {
            let shape = get_shape(0)?;
            let axes: Vec<i64> = if node.inputs.len() > 1 {
                // axes as input tensor - check constant_values
                get_constant(1).unwrap_or_default()
            } else {
                node.get_ints_attr_or("axes", &[]).to_vec()
            };

            if axes.is_empty() {
                // Remove all dimensions of size 1
                Ok(shapes_only(vec![shape
                    .into_iter()
                    .filter(|&d| d != 1)
                    .collect()]))
            } else {
                // Remove specific axes
                let rank = shape.len() as i64;
                let axes_normalized: Vec<usize> = axes
                    .iter()
                    .map(|&a| {
                        if a < 0 {
                            (rank + a) as usize
                        } else {
                            a as usize
                        }
                    })
                    .collect();
                let out_shape: Vec<u32> = shape
                    .into_iter()
                    .enumerate()
                    .filter(|(i, _)| !axes_normalized.contains(i))
                    .map(|(_, d)| d)
                    .collect();
                Ok(shapes_only(vec![out_shape]))
            }
        }

        // Concat - also propagates constant values
        "Concat" => {
            let axis = node.get_int_attr_or("axis", 0);

            // Get shapes of all inputs and try to collect constant values
            let mut first_shape: Option<Vec<u32>> = None;
            let mut total_size = 0u32;
            let mut all_constants = true;
            let mut concat_values: Vec<i64> = Vec::new();

            for (idx, input_name) in node.inputs.iter().enumerate() {
                if input_name.is_empty() {
                    continue;
                }
                if let Some(shape) = shapes.get(input_name) {
                    let rank = shape.len();
                    let axis_idx = if axis < 0 {
                        (rank as i64 + axis) as usize
                    } else {
                        axis as usize
                    };

                    if first_shape.is_none() {
                        first_shape = Some(shape.clone());
                    }
                    total_size += shape.get(axis_idx).copied().unwrap_or(1);

                    // Try to get constant values
                    if let Some(vals) = get_constant(idx) {
                        concat_values.extend(vals);
                    } else {
                        all_constants = false;
                    }
                }
            }

            if let Some(mut shape) = first_shape {
                let axis_idx = if axis < 0 {
                    (shape.len() as i64 + axis) as usize
                } else {
                    axis as usize
                };
                if axis_idx < shape.len() {
                    shape[axis_idx] = total_size;
                }
                let const_out = if all_constants && !concat_values.is_empty() {
                    Some(concat_values)
                } else {
                    None
                };
                Ok((vec![shape], vec![const_out]))
            } else {
                Ok(shapes_only(vec![vec![]]))
            }
        }

        // Conv operation
        "Conv" => {
            let input_shape = get_shape(0)?; // [N, C, H, W] or [N, C, D, H, W]
            let weight_shape = get_shape(1)?; // [M, C/group, kH, kW]

            let kernel_shape = node.get_ints_attr_or("kernel_shape", &[]);
            let strides = node.get_ints_attr_or("strides", &[1, 1]);
            let pads = node.get_ints_attr_or("pads", &[0, 0, 0, 0]);
            let dilations = node.get_ints_attr_or("dilations", &[1, 1]);
            let auto_pad = node.get_string_attr_or("auto_pad", "NOTSET");

            // Use kernel_shape from attribute or infer from weight
            let kh = if !kernel_shape.is_empty() {
                kernel_shape[0] as u32
            } else {
                weight_shape.get(2).copied().unwrap_or(1)
            };
            let kw = if kernel_shape.len() >= 2 {
                kernel_shape[1] as u32
            } else {
                weight_shape.get(3).copied().unwrap_or(1)
            };

            let sh = strides.first().copied().unwrap_or(1) as u32;
            let sw = strides.get(1).copied().unwrap_or(1) as u32;
            let dh = dilations.first().copied().unwrap_or(1) as u32;
            let dw = dilations.get(1).copied().unwrap_or(1) as u32;

            let n = input_shape.first().copied().unwrap_or(1);
            let h = input_shape.get(2).copied().unwrap_or(1);
            let w = input_shape.get(3).copied().unwrap_or(1);
            let m = weight_shape.first().copied().unwrap_or(1); // output channels

            // Handle auto_pad
            let (oh, ow) = if auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER" {
                // SAME padding: output_size = ceil(input_size / stride)
                (h.div_ceil(sh), w.div_ceil(sw))
            } else {
                // VALID or explicit padding
                let ph = pads.first().copied().unwrap_or(0) as u32;
                let pw = pads.get(1).copied().unwrap_or(0) as u32;
                // Output size = floor((input + 2*pad - dilation*(kernel-1) - 1) / stride + 1)
                (
                    (h + 2 * ph - dh * (kh - 1) - 1) / sh + 1,
                    (w + 2 * pw - dw * (kw - 1) - 1) / sw + 1,
                )
            };

            Ok(shapes_only(vec![vec![n, m, oh, ow]]))
        }

        // MaxPool / AveragePool
        "MaxPool" | "AveragePool" | "GlobalAveragePool" | "GlobalMaxPool" => {
            let input_shape = get_shape(0)?;

            if node.op_type.starts_with("Global") {
                // Global pooling reduces spatial dims to 1
                let n = input_shape.first().copied().unwrap_or(1);
                let c = input_shape.get(1).copied().unwrap_or(1);
                return Ok(shapes_only(vec![vec![n, c, 1, 1]]));
            }

            let kernel_shape = node.get_ints_attr_or("kernel_shape", &[2, 2]);
            let strides = node.get_ints_attr_or("strides", &[1, 1]);
            let pads = node.get_ints_attr_or("pads", &[0, 0, 0, 0]);
            let auto_pad = node.get_string_attr_or("auto_pad", "NOTSET");
            let ceil_mode = node.get_int_attr_or("ceil_mode", 0) != 0;

            // eprintln!("Shape inference for {}: kernel={:?}, strides={:?}, pads={:?}, auto_pad={}, ceil_mode={}", node.name, kernel_shape, strides, pads, auto_pad, ceil_mode);

            let kh = kernel_shape.first().copied().unwrap_or(2) as u32;
            let kw = kernel_shape.get(1).copied().unwrap_or(2) as u32;
            let sh = strides.first().copied().unwrap_or(1) as u32;
            let sw = strides.get(1).copied().unwrap_or(1) as u32;
            // ONNX pads format: [pad_h_begin, pad_w_begin, pad_h_end, pad_w_end]
            // Total padding = begin + end
            let ph_begin = pads.first().copied().unwrap_or(0) as u32;
            let pw_begin = pads.get(1).copied().unwrap_or(0) as u32;
            let ph_end = pads.get(2).copied().unwrap_or(0) as u32;
            let pw_end = pads.get(3).copied().unwrap_or(0) as u32;
            let ph = ph_begin + ph_end;
            let pw = pw_begin + pw_end;

            let n = input_shape.first().copied().unwrap_or(1);
            let c = input_shape.get(1).copied().unwrap_or(1);
            let h = input_shape.get(2).copied().unwrap_or(1);
            let w = input_shape.get(3).copied().unwrap_or(1);

            // Handle auto_pad and ceil_mode
            // Note: ph/pw are already total padding (begin + end)
            let (oh, ow) = if auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER" {
                // SAME padding: output_size = ceil(input_size / stride)
                (h.div_ceil(sh), w.div_ceil(sw))
            } else if ceil_mode {
                // ceil_mode: use ceiling division
                let oh = (h + ph).saturating_sub(kh).saturating_add(sh - 1) / sh + 1;
                let ow = (w + pw).saturating_sub(kw).saturating_add(sw - 1) / sw + 1;
                (oh.max(1), ow.max(1))
            } else {
                // VALID or explicit padding (floor mode)
                let oh = (h + ph).saturating_sub(kh) / sh + 1;
                let ow = (w + pw).saturating_sub(kw) / sw + 1;
                (oh.max(1), ow.max(1))
            };

            // eprintln!("  input: [{}, {}, {}, {}] -> output: [{}, {}, {}, {}]", n, c, h, w, n, c, oh, ow);

            // MaxPool can have optional indices output
            let mut outputs = vec![vec![n, c, oh, ow]];
            if node.outputs.len() > 1 {
                outputs.push(vec![n, c, oh, ow]); // indices have same shape
            }
            Ok(shapes_only(outputs))
        }

        // Reduce operations
        "ReduceSum" | "ReduceMean" | "ReduceMax" | "ReduceMin" | "ReduceProd" => {
            let shape = get_shape(0)?;
            let axes = node.get_ints_attr_or("axes", &[]);
            let keepdims = node.get_int_attr_or("keepdims", 1) != 0;

            // If no axes specified, reduce all
            let axes: Vec<i64> = if axes.is_empty() {
                (0..shape.len() as i64).collect()
            } else {
                axes.to_vec()
            };

            // Normalize negative axes
            let normalized_axes: Vec<usize> = axes
                .iter()
                .map(|&a| {
                    if a < 0 {
                        (shape.len() as i64 + a) as usize
                    } else {
                        a as usize
                    }
                })
                .collect();

            let mut out_shape: Vec<u32> = Vec::new();
            for (i, &dim) in shape.iter().enumerate() {
                if normalized_axes.contains(&i) {
                    if keepdims {
                        out_shape.push(1);
                    }
                } else {
                    out_shape.push(dim);
                }
            }
            if out_shape.is_empty() {
                out_shape.push(1); // Scalar output
            }
            Ok(shapes_only(vec![out_shape]))
        }

        _ => {
            // Unknown op - return empty shapes (will fail at execution time)
            let n = node.outputs.len();
            Ok((node.outputs.iter().map(|_| vec![]).collect(), vec![None; n]))
        }
    }
}

/// Compute broadcast output shape.
fn broadcast_shapes(a: &[u32], b: &[u32]) -> Result<Vec<u32>, OnnxError> {
    let max_rank = a.len().max(b.len());
    let mut result = vec![1u32; max_rank];

    for i in 0..max_rank {
        let a_dim = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let b_dim = if i < b.len() { b[b.len() - 1 - i] } else { 1 };

        if a_dim == b_dim {
            result[max_rank - 1 - i] = a_dim;
        } else if a_dim == 1 {
            result[max_rank - 1 - i] = b_dim;
        } else if b_dim == 1 {
            result[max_rank - 1 - i] = a_dim;
        } else {
            return Err(OnnxError::ShapeInferenceError {
                node: "broadcast".to_string(),
                reason: format!(
                    "Cannot broadcast shapes {:?} and {:?}: incompatible dimensions {} and {}",
                    a, b, a_dim, b_dim
                ),
            });
        }
    }

    Ok(result)
}

/// Compute matmul output shape.
fn matmul_shape(a: &[u32], b: &[u32]) -> Result<Vec<u32>, OnnxError> {
    if a.is_empty() || b.is_empty() {
        return Err(OnnxError::ShapeInferenceError {
            node: "matmul".to_string(),
            reason: "MatMul requires non-scalar inputs".to_string(),
        });
    }

    // Handle 1D inputs
    let (a_shape, a_prepend) = if a.len() == 1 {
        (vec![1, a[0]], true)
    } else {
        (a.to_vec(), false)
    };
    let (b_shape, b_append) = if b.len() == 1 {
        (vec![b[0], 1], true)
    } else {
        (b.to_vec(), false)
    };

    let a_rank = a_shape.len();
    let b_rank = b_shape.len();

    // Get matrix dimensions
    let a_rows = a_shape[a_rank - 2];
    let a_cols = a_shape[a_rank - 1];
    let b_rows = b_shape[b_rank - 2];
    let b_cols = b_shape[b_rank - 1];

    if a_cols != b_rows {
        return Err(OnnxError::ShapeInferenceError {
            node: "matmul".to_string(),
            reason: format!(
                "MatMul dimension mismatch: ({}, {}) x ({}, {})",
                a_rows, a_cols, b_rows, b_cols
            ),
        });
    }

    // Broadcast batch dimensions
    let a_batch = &a_shape[..a_rank - 2];
    let b_batch = &b_shape[..b_rank - 2];
    let out_batch = broadcast_shapes(a_batch, b_batch)?;

    // Build output shape
    let mut out_shape = out_batch;
    if !a_prepend {
        out_shape.push(a_rows);
    }
    if !b_append {
        out_shape.push(b_cols);
    }

    // Handle case where both were 1D (scalar output)
    if out_shape.is_empty() {
        out_shape.push(1);
    }

    Ok(out_shape)
}
