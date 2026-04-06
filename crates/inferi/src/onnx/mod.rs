//! ONNX model loading and execution.
//!
//! This module provides functionality to load ONNX models and execute them on GPU
//! using the existing vortx/khal infrastructure.
//!
//! # Example
//!
//! ```ignore
//! use inferi::onnx::OnnxModel;
//! use std::collections::HashMap;
//!
//! // Load model
//! let model = OnnxModel::from_file("model.onnx")?;
//!
//! // Compile for GPU with input shapes
//! let mut input_shapes = HashMap::new();
//! input_shapes.insert("input".to_string(), vec![1, 3, 224, 224]);
//! let compiled = model.compile(&backend, &input_shapes)?;
//!
//! // Run inference
//! let mut inputs = HashMap::new();
//! inputs.insert("input".to_string(), &input_tensor);
//! let outputs = compiled.run(&mut ctxt, inputs).await?;
//! ```
//!
//! # Supported Operations
//!
//! ## Unary Operations
//! - Relu, Sigmoid, Tanh, Gelu, Silu
//! - Abs, Neg, Sqrt, Log, Sin, Cos
//! - Elu, LeakyRelu, HardSigmoid
//! - Exp, Reciprocal, Erf, Clip, Pow
//!
//! ## Binary Operations
//! - Add, Sub, Mul, Div (with broadcasting)
//!
//! ## Matrix Operations
//! - MatMul, Gemm
//!
//! ## Normalization
//! - Softmax, LayerNormalization
//!
//! ## Shape Operations
//! - Reshape, Transpose, Squeeze, Unsqueeze, Flatten, Identity
//! - Gather, Concat
//!
//! ## Reduction Operations
//! - ReduceSum, ReduceMean, ReduceMax, ReduceMin
//!
//! ## Constants
//! - Constant (tensor and scalar)

mod error;
mod graph;
mod model;
mod ops;
mod tensor;

pub use error::OnnxError;
pub use graph::{AttributeValue, GraphInput, OnnxGraph, OnnxNode};
pub use model::{CompiledOnnxGraph, OnnxModel};
