//! Utilities for converting ONNX TensorProto to vortx `Tensor<f32>`.

use crate::onnx::error::OnnxError;
use half::{bf16, f16};
use onnx_protobuf::{tensor_proto::DataType, TensorProto};
use protobuf::Enum;

/// Extracts f32 data from a TensorProto, converting from the native data type.
///
/// All computations in the ONNX runtime are done in f32, so f16/bf16 data is
/// dequantized on load.
pub fn tensor_proto_to_f32_data(proto: &TensorProto) -> Result<Vec<f32>, OnnxError> {
    let data_type = DataType::from_i32(proto.data_type)
        .ok_or_else(|| OnnxError::UnsupportedDataType(proto.data_type, proto.name.clone()))?;

    // Calculate expected number of elements from dims
    let numel: usize = proto.dims.iter().map(|&d| d as usize).product();
    if numel == 0 {
        return Ok(Vec::new());
    }

    // First check if data is in raw_data field
    if !proto.raw_data.is_empty() {
        return extract_from_raw_data(&proto.raw_data, data_type, numel, &proto.name);
    }

    // Check if data is in external_data (external file reference)
    if !proto.external_data.is_empty() {
        return Err(OnnxError::ParseError(format!(
            "Tensor '{}' uses external data which is not yet supported",
            proto.name
        )));
    }

    // Otherwise, check type-specific fields
    match data_type {
        DataType::FLOAT => {
            if proto.float_data.len() >= numel {
                Ok(proto.float_data[..numel].to_vec())
            } else {
                Err(OnnxError::ParseError(format!(
                    "Tensor '{}' has {} elements but float_data has only {}",
                    proto.name,
                    numel,
                    proto.float_data.len()
                )))
            }
        }
        DataType::DOUBLE => {
            if proto.double_data.len() >= numel {
                Ok(proto.double_data[..numel]
                    .iter()
                    .map(|&x| x as f32)
                    .collect())
            } else {
                Err(OnnxError::ParseError(format!(
                    "Tensor '{}' has {} elements but double_data has only {}",
                    proto.name,
                    numel,
                    proto.double_data.len()
                )))
            }
        }
        DataType::INT32 => {
            if proto.int32_data.len() >= numel {
                Ok(proto.int32_data[..numel]
                    .iter()
                    .map(|&x| x as f32)
                    .collect())
            } else {
                Err(OnnxError::ParseError(format!(
                    "Tensor '{}' has {} elements but int32_data has only {}",
                    proto.name,
                    numel,
                    proto.int32_data.len()
                )))
            }
        }
        DataType::INT64 => {
            if proto.int64_data.len() >= numel {
                Ok(proto.int64_data[..numel]
                    .iter()
                    .map(|&x| x as f32)
                    .collect())
            } else {
                Err(OnnxError::ParseError(format!(
                    "Tensor '{}' has {} elements but int64_data has only {}",
                    proto.name,
                    numel,
                    proto.int64_data.len()
                )))
            }
        }
        DataType::UINT64 => {
            if proto.uint64_data.len() >= numel {
                Ok(proto.uint64_data[..numel]
                    .iter()
                    .map(|&x| x as f32)
                    .collect())
            } else {
                Err(OnnxError::ParseError(format!(
                    "Tensor '{}' has {} elements but uint64_data has only {}",
                    proto.name,
                    numel,
                    proto.uint64_data.len()
                )))
            }
        }
        // For types stored in int32_data (FLOAT16, UINT8, INT8, etc.), use raw_data path
        _ => Err(OnnxError::ParseError(format!(
            "Tensor '{}' has data_type {:?} but no raw_data field",
            proto.name, data_type
        ))),
    }
}

/// Extract f32 data from raw_data bytes based on data type.
fn extract_from_raw_data(
    raw_data: &[u8],
    data_type: DataType,
    numel: usize,
    name: &str,
) -> Result<Vec<f32>, OnnxError> {
    match data_type {
        DataType::FLOAT => {
            let expected_bytes = numel * 4;
            if raw_data.len() < expected_bytes {
                return Err(OnnxError::ParseError(format!(
                    "Tensor '{}' expects {} bytes but raw_data has {}",
                    name,
                    expected_bytes,
                    raw_data.len()
                )));
            }
            let floats: &[f32] = bytemuck::cast_slice(&raw_data[..expected_bytes]);
            Ok(floats.to_vec())
        }
        DataType::DOUBLE => {
            let expected_bytes = numel * 8;
            if raw_data.len() < expected_bytes {
                return Err(OnnxError::ParseError(format!(
                    "Tensor '{}' expects {} bytes but raw_data has {}",
                    name,
                    expected_bytes,
                    raw_data.len()
                )));
            }
            let doubles: &[f64] = bytemuck::cast_slice(&raw_data[..expected_bytes]);
            Ok(doubles.iter().map(|&x| x as f32).collect())
        }
        DataType::FLOAT16 => {
            let expected_bytes = numel * 2;
            if raw_data.len() < expected_bytes {
                return Err(OnnxError::ParseError(format!(
                    "Tensor '{}' expects {} bytes but raw_data has {}",
                    name,
                    expected_bytes,
                    raw_data.len()
                )));
            }
            let halfs: &[u16] = bytemuck::cast_slice(&raw_data[..expected_bytes]);
            Ok(halfs
                .iter()
                .map(|&bits| f16::from_bits(bits).to_f32())
                .collect())
        }
        DataType::BFLOAT16 => {
            let expected_bytes = numel * 2;
            if raw_data.len() < expected_bytes {
                return Err(OnnxError::ParseError(format!(
                    "Tensor '{}' expects {} bytes but raw_data has {}",
                    name,
                    expected_bytes,
                    raw_data.len()
                )));
            }
            let halfs: &[u16] = bytemuck::cast_slice(&raw_data[..expected_bytes]);
            Ok(halfs
                .iter()
                .map(|&bits| bf16::from_bits(bits).to_f32())
                .collect())
        }
        DataType::INT8 => {
            if raw_data.len() < numel {
                return Err(OnnxError::ParseError(format!(
                    "Tensor '{}' expects {} bytes but raw_data has {}",
                    name,
                    numel,
                    raw_data.len()
                )));
            }
            let i8s: &[i8] = bytemuck::cast_slice(&raw_data[..numel]);
            Ok(i8s.iter().map(|&x| x as f32).collect())
        }
        DataType::UINT8 => {
            if raw_data.len() < numel {
                return Err(OnnxError::ParseError(format!(
                    "Tensor '{}' expects {} bytes but raw_data has {}",
                    name,
                    numel,
                    raw_data.len()
                )));
            }
            Ok(raw_data[..numel].iter().map(|&x| x as f32).collect())
        }
        DataType::INT16 => {
            let expected_bytes = numel * 2;
            if raw_data.len() < expected_bytes {
                return Err(OnnxError::ParseError(format!(
                    "Tensor '{}' expects {} bytes but raw_data has {}",
                    name,
                    expected_bytes,
                    raw_data.len()
                )));
            }
            let i16s: &[i16] = bytemuck::cast_slice(&raw_data[..expected_bytes]);
            Ok(i16s.iter().map(|&x| x as f32).collect())
        }
        DataType::UINT16 => {
            let expected_bytes = numel * 2;
            if raw_data.len() < expected_bytes {
                return Err(OnnxError::ParseError(format!(
                    "Tensor '{}' expects {} bytes but raw_data has {}",
                    name,
                    expected_bytes,
                    raw_data.len()
                )));
            }
            let u16s: &[u16] = bytemuck::cast_slice(&raw_data[..expected_bytes]);
            Ok(u16s.iter().map(|&x| x as f32).collect())
        }
        DataType::INT32 => {
            let expected_bytes = numel * 4;
            if raw_data.len() < expected_bytes {
                return Err(OnnxError::ParseError(format!(
                    "Tensor '{}' expects {} bytes but raw_data has {}",
                    name,
                    expected_bytes,
                    raw_data.len()
                )));
            }
            let i32s: &[i32] = bytemuck::cast_slice(&raw_data[..expected_bytes]);
            Ok(i32s.iter().map(|&x| x as f32).collect())
        }
        DataType::UINT32 => {
            let expected_bytes = numel * 4;
            if raw_data.len() < expected_bytes {
                return Err(OnnxError::ParseError(format!(
                    "Tensor '{}' expects {} bytes but raw_data has {}",
                    name,
                    expected_bytes,
                    raw_data.len()
                )));
            }
            let u32s: &[u32] = bytemuck::cast_slice(&raw_data[..expected_bytes]);
            Ok(u32s.iter().map(|&x| x as f32).collect())
        }
        DataType::INT64 => {
            let expected_bytes = numel * 8;
            if raw_data.len() < expected_bytes {
                return Err(OnnxError::ParseError(format!(
                    "Tensor '{}' expects {} bytes but raw_data has {}",
                    name,
                    expected_bytes,
                    raw_data.len()
                )));
            }
            let i64s: &[i64] = bytemuck::cast_slice(&raw_data[..expected_bytes]);
            Ok(i64s.iter().map(|&x| x as f32).collect())
        }
        DataType::UINT64 => {
            let expected_bytes = numel * 8;
            if raw_data.len() < expected_bytes {
                return Err(OnnxError::ParseError(format!(
                    "Tensor '{}' expects {} bytes but raw_data has {}",
                    name,
                    expected_bytes,
                    raw_data.len()
                )));
            }
            let u64s: &[u64] = bytemuck::cast_slice(&raw_data[..expected_bytes]);
            Ok(u64s.iter().map(|&x| x as f32).collect())
        }
        DataType::BOOL => {
            if raw_data.len() < numel {
                return Err(OnnxError::ParseError(format!(
                    "Tensor '{}' expects {} bytes but raw_data has {}",
                    name,
                    numel,
                    raw_data.len()
                )));
            }
            Ok(raw_data[..numel]
                .iter()
                .map(|&x| if x != 0 { 1.0 } else { 0.0 })
                .collect())
        }
        _ => Err(OnnxError::UnsupportedDataType(
            data_type as i32,
            name.to_string(),
        )),
    }
}

/// Convert TensorProto dimensions to u32 shape.
pub fn tensor_proto_shape(proto: &TensorProto) -> Vec<u32> {
    proto.dims.iter().map(|&d| d as u32).collect()
}

/// Extract i64 data from a TensorProto (for indices, shapes, etc.).
pub fn tensor_proto_to_i64_data(proto: &TensorProto) -> Result<Vec<i64>, OnnxError> {
    let data_type = DataType::from_i32(proto.data_type)
        .ok_or_else(|| OnnxError::UnsupportedDataType(proto.data_type, proto.name.clone()))?;

    let numel: usize = proto.dims.iter().map(|&d| d as usize).product();
    if numel == 0 {
        return Ok(Vec::new());
    }

    // Check raw_data first
    if !proto.raw_data.is_empty() {
        match data_type {
            DataType::INT64 => {
                let expected_bytes = numel * 8;
                if proto.raw_data.len() >= expected_bytes {
                    let i64s: &[i64] = bytemuck::cast_slice(&proto.raw_data[..expected_bytes]);
                    return Ok(i64s.to_vec());
                }
            }
            DataType::INT32 => {
                let expected_bytes = numel * 4;
                if proto.raw_data.len() >= expected_bytes {
                    let i32s: &[i32] = bytemuck::cast_slice(&proto.raw_data[..expected_bytes]);
                    return Ok(i32s.iter().map(|&x| x as i64).collect());
                }
            }
            _ => {}
        }
    }

    // Check type-specific fields
    match data_type {
        DataType::INT64 => {
            if proto.int64_data.len() >= numel {
                Ok(proto.int64_data[..numel].to_vec())
            } else {
                Err(OnnxError::ParseError(format!(
                    "Tensor '{}' has {} elements but int64_data has only {}",
                    proto.name,
                    numel,
                    proto.int64_data.len()
                )))
            }
        }
        DataType::INT32 => {
            if proto.int32_data.len() >= numel {
                Ok(proto.int32_data[..numel]
                    .iter()
                    .map(|&x| x as i64)
                    .collect())
            } else {
                Err(OnnxError::ParseError(format!(
                    "Tensor '{}' has {} elements but int32_data has only {}",
                    proto.name,
                    numel,
                    proto.int32_data.len()
                )))
            }
        }
        _ => Err(OnnxError::UnsupportedDataType(
            proto.data_type,
            proto.name.clone(),
        )),
    }
}
