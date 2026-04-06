//! ONNX-specific error types.

use khal::backend::GpuBackendError;
use thiserror::Error;

/// Errors that can occur when loading or executing ONNX models.
#[derive(Error, Debug)]
pub enum OnnxError {
    /// Error reading or parsing the ONNX protobuf file.
    #[error("Failed to parse ONNX model: {0}")]
    ParseError(String),

    /// I/O error when reading model file.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Missing required input tensor.
    #[error("Missing input tensor: {0}")]
    MissingInput(String),

    /// Shape mismatch between expected and provided input.
    #[error("Shape mismatch for '{name}': expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        name: String,
        expected: Vec<u32>,
        actual: Vec<u32>,
    },

    /// Unsupported ONNX operation.
    #[error("Unsupported operation '{op}' at node '{node}'")]
    UnsupportedOp { op: String, node: String },

    /// Unsupported data type.
    #[error("Unsupported data type {0} in tensor '{1}'")]
    UnsupportedDataType(i32, String),

    /// Invalid attribute value.
    #[error("Invalid attribute '{attr}' in node '{node}': {reason}")]
    InvalidAttribute {
        attr: String,
        node: String,
        reason: String,
    },

    /// Shape inference failed.
    #[error("Shape inference failed for node '{node}': {reason}")]
    ShapeInferenceError { node: String, reason: String },

    /// Graph has a cycle (not a DAG).
    #[error("Graph contains a cycle involving node '{0}'")]
    CyclicGraph(String),

    /// GPU backend error during execution.
    #[error("GPU error: {0}")]
    GpuError(#[from] GpuBackendError),

    /// Internal error (should not happen).
    #[error("Internal error: {0}")]
    InternalError(String),
}

impl From<protobuf::Error> for OnnxError {
    fn from(err: protobuf::Error) -> Self {
        OnnxError::ParseError(err.to_string())
    }
}
