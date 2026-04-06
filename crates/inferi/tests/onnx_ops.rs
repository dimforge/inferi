//! Tests for ONNX operations.

#![cfg(feature = "onnx")]

use inferi::context::{LlmContext, LlmOps};
use inferi::tensor_cache::TensorCache;
#[cfg(feature = "cuda")]
use khal::backend::Cuda;
use khal::backend::{Backend, GpuBackend, WebGpu};
use khal::BufferUsages;
use std::sync::Arc;
use vortx::shapes::TensorLayoutBuffers;
use vortx::tensor::TensorBuilder;
use wgpu::{Features, Limits};

async fn create_webgpu_backend() -> Arc<GpuBackend> {
    let webgpu = WebGpu::new(Features::default(), Limits::default())
        .await
        .unwrap();
    Arc::new(GpuBackend::WebGpu(webgpu))
}

#[cfg(feature = "cpu")]
fn cpu_backend() -> Arc<GpuBackend> {
    Arc::new(GpuBackend::Cpu)
}

#[cfg(feature = "cuda")]
fn cuda_backend() -> Arc<GpuBackend> {
    Arc::new(GpuBackend::Cuda(Cuda::new(0).unwrap()))
}

fn create_context<'a>(
    backend: &'a GpuBackend,
    ops: &'a LlmOps,
    cache: &'a mut TensorCache,
    shapes: &'a mut TensorLayoutBuffers,
) -> LlmContext<'a> {
    LlmContext {
        backend,
        cache,
        shapes,
        pass: None,
        encoder: None,
        ops,
    }
}

// =============================================================================
// test_unary_ops
// =============================================================================

async fn test_unary_ops_generic(backend: &Arc<GpuBackend>) {
    let ops = Arc::new(LlmOps::new(backend).unwrap());
    let mut cache = TensorCache::default();
    let mut shapes = TensorLayoutBuffers::new(backend);

    let mut ctxt = create_context(backend, &ops, &mut cache, &mut shapes);

    // Create input tensor [2, 3]
    let data = vec![1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0];
    let input = TensorBuilder::tensor(&[2, 3], BufferUsages::STORAGE)
        .build_init(backend, &data)
        .unwrap();

    // Test Relu
    ctxt.begin_submission();
    let result = ctxt.unop(inferi::ops::UnaryOp::Relu, &input).unwrap();
    ctxt.submit();
    backend.synchronize().unwrap();

    let mut output = vec![0.0f32; 6];
    backend
        .slow_read_buffer(result.buffer(), &mut output)
        .await
        .unwrap();

    assert_eq!(output, vec![1.0, 0.0, 3.0, 0.0, 5.0, 0.0]);
    println!("Relu test passed: {:?}", output);

    // Test Abs
    ctxt.begin_submission();
    let result = ctxt.unop(inferi::ops::UnaryOp::Abs, &input).unwrap();
    ctxt.submit();
    backend.synchronize().unwrap();

    backend
        .slow_read_buffer(result.buffer(), &mut output)
        .await
        .unwrap();
    assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    println!("Abs test passed: {:?}", output);

    // Test Exp
    ctxt.begin_submission();
    let small_data = vec![0.0f32, 1.0, 2.0];
    let small_input = TensorBuilder::tensor(&[3], BufferUsages::STORAGE)
        .build_init(backend, &small_data)
        .unwrap();
    let result = ctxt.unop(inferi::ops::UnaryOp::Exp, &small_input).unwrap();
    ctxt.submit();
    backend.synchronize().unwrap();

    let mut small_output = vec![0.0f32; 3];
    backend
        .slow_read_buffer(result.buffer(), &mut small_output)
        .await
        .unwrap();

    // Check exp values with tolerance
    let expected = [
        1.0f32,
        std::f32::consts::E,
        std::f32::consts::E * std::f32::consts::E,
    ];
    for (a, b) in small_output.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 0.001, "Exp mismatch: {} vs {}", a, b);
    }
    println!("Exp test passed: {:?}", small_output);
}

#[async_std::test]
#[serial_test::serial]
async fn test_unary_ops_webgpu() {
    let backend = create_webgpu_backend().await;
    test_unary_ops_generic(&backend).await;
}

#[cfg(feature = "cpu")]
#[async_std::test]
async fn test_unary_ops_cpu() {
    let backend = cpu_backend();
    test_unary_ops_generic(&backend).await;
}

#[cfg(feature = "cuda")]
#[async_std::test]
#[serial_test::serial]
async fn test_unary_ops_cuda() {
    let backend = cuda_backend();
    test_unary_ops_generic(&backend).await;
}

// =============================================================================
// test_binary_ops
// =============================================================================

async fn test_binary_ops_generic(backend: &Arc<GpuBackend>) {
    let ops = Arc::new(LlmOps::new(backend).unwrap());
    let mut cache = TensorCache::default();
    let mut shapes = TensorLayoutBuffers::new(backend);

    let mut ctxt = create_context(backend, &ops, &mut cache, &mut shapes);

    // Create input tensors [2, 2]
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let b_data = vec![5.0f32, 6.0, 7.0, 8.0];

    let a = TensorBuilder::tensor(&[2, 2], BufferUsages::STORAGE)
        .build_init(backend, &a_data)
        .unwrap();
    let b = TensorBuilder::tensor(&[2, 2], BufferUsages::STORAGE)
        .build_init(backend, &b_data)
        .unwrap();

    // Test Add
    ctxt.begin_submission();
    let result = ctxt.add(&a, &b).unwrap();
    ctxt.submit();
    backend.synchronize().unwrap();

    let mut output = vec![0.0f32; 4];
    backend
        .slow_read_buffer(result.buffer(), &mut output)
        .await
        .unwrap();
    assert_eq!(output, vec![6.0, 8.0, 10.0, 12.0]);
    println!("Add test passed: {:?}", output);

    // Test Mul
    ctxt.begin_submission();
    let result = ctxt.mul(&a, &b).unwrap();
    ctxt.submit();
    backend.synchronize().unwrap();

    backend
        .slow_read_buffer(result.buffer(), &mut output)
        .await
        .unwrap();
    assert_eq!(output, vec![5.0, 12.0, 21.0, 32.0]);
    println!("Mul test passed: {:?}", output);
}

#[async_std::test]
#[serial_test::serial]
async fn test_binary_ops_webgpu() {
    let backend = create_webgpu_backend().await;
    test_binary_ops_generic(&backend).await;
}

#[cfg(feature = "cpu")]
#[async_std::test]
async fn test_binary_ops_cpu() {
    let backend = cpu_backend();
    test_binary_ops_generic(&backend).await;
}

#[cfg(feature = "cuda")]
#[async_std::test]
#[serial_test::serial]
async fn test_binary_ops_cuda() {
    let backend = cuda_backend();
    test_binary_ops_generic(&backend).await;
}

// =============================================================================
// test_matmul
// =============================================================================

async fn test_matmul_generic(backend: &Arc<GpuBackend>) {
    let ops = Arc::new(LlmOps::new(backend).unwrap());
    let mut cache = TensorCache::default();
    let mut shapes = TensorLayoutBuffers::new(backend);

    let mut ctxt = create_context(backend, &ops, &mut cache, &mut shapes);

    // Matrix A [2, 3]
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    // Matrix B [3, 2]
    let b_data = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];

    let a = TensorBuilder::tensor(&[2, 3], BufferUsages::STORAGE)
        .build_init(backend, &a_data)
        .unwrap();
    let b = TensorBuilder::tensor(&[3, 2], BufferUsages::STORAGE)
        .build_init(backend, &b_data)
        .unwrap();

    ctxt.begin_submission();
    let result = ctxt.matmul(&a, &b).unwrap();
    ctxt.submit();
    backend.synchronize().unwrap();

    // Expected: [2, 2]
    // [1,2,3] @ [7,8; 9,10; 11,12] = [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
    // [4,5,6] @ [7,8; 9,10; 11,12] = [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
    let mut output = vec![0.0f32; 4];
    backend
        .slow_read_buffer(result.buffer(), &mut output)
        .await
        .unwrap();

    let expected = [58.0f32, 64.0, 139.0, 154.0];
    for (a, b) in output.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 0.01, "MatMul mismatch: {} vs {}", a, b);
    }
    println!("MatMul test passed: {:?}", output);
}

#[async_std::test]
#[serial_test::serial]
async fn test_matmul_webgpu() {
    let backend = create_webgpu_backend().await;
    test_matmul_generic(&backend).await;
}

#[cfg(feature = "cpu")]
#[async_std::test]
async fn test_matmul_cpu() {
    let backend = cpu_backend();
    test_matmul_generic(&backend).await;
}

#[cfg(feature = "cuda")]
#[async_std::test]
#[serial_test::serial]
async fn test_matmul_cuda() {
    let backend = cuda_backend();
    test_matmul_generic(&backend).await;
}

// =============================================================================
// test_reduce_sum
// =============================================================================

async fn test_reduce_sum_generic(backend: &Arc<GpuBackend>) {
    let ops = Arc::new(LlmOps::new(backend).unwrap());
    let mut cache = TensorCache::default();
    let mut shapes = TensorLayoutBuffers::new(backend);

    let mut ctxt = create_context(backend, &ops, &mut cache, &mut shapes);

    // Create input tensor [2, 3]
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input = TensorBuilder::tensor(&[2, 3], BufferUsages::STORAGE)
        .build_init(backend, &data)
        .unwrap();

    // Reduce sum along axis 1 (columns) -> [2, 1]
    ctxt.begin_submission();
    let result = ctxt.reduce_sum_axis(&input, 1, &[2, 1]).unwrap();
    ctxt.submit();
    backend.synchronize().unwrap();

    let mut output = vec![0.0f32; 2];
    backend
        .slow_read_buffer(result.buffer(), &mut output)
        .await
        .unwrap();

    // Row 0: 1+2+3 = 6, Row 1: 4+5+6 = 15
    let expected = [6.0f32, 15.0];
    for (a, b) in output.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 0.01, "ReduceSum mismatch: {} vs {}", a, b);
    }
    println!("ReduceSum test passed: {:?}", output);
}

#[async_std::test]
#[serial_test::serial]
async fn test_reduce_sum_webgpu() {
    let backend = create_webgpu_backend().await;
    test_reduce_sum_generic(&backend).await;
}

#[cfg(feature = "cpu")]
#[async_std::test]
async fn test_reduce_sum_cpu() {
    let backend = cpu_backend();
    test_reduce_sum_generic(&backend).await;
}

#[cfg(feature = "cuda")]
#[async_std::test]
#[serial_test::serial]
async fn test_reduce_sum_cuda() {
    let backend = cuda_backend();
    test_reduce_sum_generic(&backend).await;
}

// =============================================================================
// test_reduce_mean
// =============================================================================

async fn test_reduce_mean_generic(backend: &Arc<GpuBackend>) {
    let ops = Arc::new(LlmOps::new(backend).unwrap());
    let mut cache = TensorCache::default();
    let mut shapes = TensorLayoutBuffers::new(backend);

    let mut ctxt = create_context(backend, &ops, &mut cache, &mut shapes);

    // Create input tensor [2, 3]
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input = TensorBuilder::tensor(&[2, 3], BufferUsages::STORAGE)
        .build_init(backend, &data)
        .unwrap();

    // Reduce mean along axis 1 (columns) -> [2, 1]
    ctxt.begin_submission();
    let result = ctxt.reduce_mean_axis(&input, 1, &[2, 1]).unwrap();
    ctxt.submit();
    backend.synchronize().unwrap();

    let mut output = vec![0.0f32; 2];
    backend
        .slow_read_buffer(result.buffer(), &mut output)
        .await
        .unwrap();

    // Row 0: (1+2+3)/3 = 2, Row 1: (4+5+6)/3 = 5
    let expected = [2.0f32, 5.0];
    for (a, b) in output.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 0.01, "ReduceMean mismatch: {} vs {}", a, b);
    }
    println!("ReduceMean test passed: {:?}", output);
}

#[async_std::test]
#[serial_test::serial]
async fn test_reduce_mean_webgpu() {
    let backend = create_webgpu_backend().await;
    test_reduce_mean_generic(&backend).await;
}

#[cfg(feature = "cpu")]
#[async_std::test]
async fn test_reduce_mean_cpu() {
    let backend = cpu_backend();
    test_reduce_mean_generic(&backend).await;
}

#[cfg(feature = "cuda")]
#[async_std::test]
#[serial_test::serial]
async fn test_reduce_mean_cuda() {
    let backend = cuda_backend();
    test_reduce_mean_generic(&backend).await;
}

// =============================================================================
// test_onnx_mlp_layer
// =============================================================================

/// Test the full ONNX model compile/run pipeline by simulating a simple
/// matmul + add + relu graph (like a simple MLP layer).
async fn test_onnx_mlp_layer_generic(backend: &Arc<GpuBackend>) {
    let ops = Arc::new(LlmOps::new(backend).unwrap());
    let mut cache = TensorCache::default();
    let mut shapes = TensorLayoutBuffers::new(backend);

    let mut ctxt = create_context(backend, &ops, &mut cache, &mut shapes);

    // Simulate what an MLP layer does: output = relu(input @ weight + bias)
    // Input: [1, 4], Weight: [4, 2], Bias: [2], Output: [1, 2]

    // Create input tensor
    let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let input = TensorBuilder::tensor(&[1, 4], BufferUsages::STORAGE)
        .build_init(backend, &input_data)
        .unwrap();

    // Create weight tensor (4x2)
    let weight_data = vec![
        0.1f32, 0.2, // row 0
        0.1, 0.2, // row 1
        0.1, 0.2, // row 2
        0.1, 0.2, // row 3
    ];
    let weight = TensorBuilder::tensor(&[4, 2], BufferUsages::STORAGE)
        .build_init(backend, &weight_data)
        .unwrap();

    // Create bias tensor (2)
    let bias_data = vec![0.5f32, -0.5];
    let bias = TensorBuilder::tensor(&[2], BufferUsages::STORAGE)
        .build_init(backend, &bias_data)
        .unwrap();

    // Run the operations manually (simulating what ONNX execution would do)
    ctxt.begin_submission();

    // Step 1: matmul(input, weight) -> [1, 2]
    // [1,2,3,4] @ [[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2]]
    // = [1*0.1+2*0.1+3*0.1+4*0.1, 1*0.2+2*0.2+3*0.2+4*0.2]
    // = [1.0, 2.0]
    let mm_result = ctxt.matmul(&input, &weight).unwrap();

    // Step 2: add(mm_result, bias) -> [1, 2]
    // [1.0, 2.0] + [0.5, -0.5] = [1.5, 1.5]
    let add_result = ctxt.add(&mm_result, &bias).unwrap();

    // Step 3: relu(add_result) -> [1, 2]
    // relu([1.5, 1.5]) = [1.5, 1.5]
    let relu_result = ctxt.unop(inferi::ops::UnaryOp::Relu, &add_result).unwrap();

    ctxt.submit();
    backend.synchronize().unwrap();

    // Read result
    let mut output = vec![0.0f32; 2];
    backend
        .slow_read_buffer(relu_result.buffer(), &mut output)
        .await
        .unwrap();

    let expected = [1.5f32, 1.5];
    for (a, b) in output.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 0.01, "MLP layer mismatch: {} vs {}", a, b);
    }
    println!("MLP layer test passed: {:?}", output);
}

#[async_std::test]
#[serial_test::serial]
async fn test_onnx_mlp_layer_webgpu() {
    let backend = create_webgpu_backend().await;
    test_onnx_mlp_layer_generic(&backend).await;
}

#[cfg(feature = "cpu")]
#[async_std::test]
async fn test_onnx_mlp_layer_cpu() {
    let backend = cpu_backend();
    test_onnx_mlp_layer_generic(&backend).await;
}

#[cfg(feature = "cuda")]
#[async_std::test]
#[serial_test::serial]
async fn test_onnx_mlp_layer_cuda() {
    let backend = cuda_backend();
    test_onnx_mlp_layer_generic(&backend).await;
}

// =============================================================================
// test_gemm
// =============================================================================

/// Test Gemm operation (commonly used in ONNX for linear layers)
async fn test_gemm_generic(backend: &Arc<GpuBackend>) {
    let ops = Arc::new(LlmOps::new(backend).unwrap());
    let mut cache = TensorCache::default();
    let mut shapes = TensorLayoutBuffers::new(backend);

    let mut ctxt = create_context(backend, &ops, &mut cache, &mut shapes);

    // Test Gemm: Y = alpha * A @ B + beta * C
    // A: [2, 3], B: [3, 2], C: [2], alpha=1, beta=1
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let c_data = vec![10.0f32, 20.0];

    let a = TensorBuilder::tensor(&[2, 3], BufferUsages::STORAGE)
        .build_init(backend, &a_data)
        .unwrap();
    let b = TensorBuilder::tensor(&[3, 2], BufferUsages::STORAGE)
        .build_init(backend, &b_data)
        .unwrap();
    let c = TensorBuilder::tensor(&[2], BufferUsages::STORAGE)
        .build_init(backend, &c_data)
        .unwrap();

    ctxt.begin_submission();

    // A @ B:
    // [1,2,3] @ [[1,2],[3,4],[5,6]] = [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
    // [4,5,6] @ [[1,2],[3,4],[5,6]] = [4*1+5*3+6*5, 4*2+5*4+6*6] = [49, 64]
    // Result: [[22,28],[49,64]]
    let mut mm_result = ctxt.matmul(&a, &b).unwrap();

    // Add bias C (broadcasts)
    // [[22,28],[49,64]] + [10,20] = [[32,48],[59,84]]
    ctxt.add_assign(&mut mm_result, &c).unwrap();

    ctxt.submit();
    backend.synchronize().unwrap();

    let mut output = vec![0.0f32; 4];
    backend
        .slow_read_buffer(mm_result.buffer(), &mut output)
        .await
        .unwrap();

    let expected = [32.0f32, 48.0, 59.0, 84.0];
    for (a, b) in output.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 0.01, "Gemm mismatch: {} vs {}", a, b);
    }
    println!("Gemm test passed: {:?}", output);
}

#[async_std::test]
#[serial_test::serial]
async fn test_gemm_webgpu() {
    let backend = create_webgpu_backend().await;
    test_gemm_generic(&backend).await;
}

#[cfg(feature = "cpu")]
#[async_std::test]
async fn test_gemm_cpu() {
    let backend = cpu_backend();
    test_gemm_generic(&backend).await;
}

#[cfg(feature = "cuda")]
#[async_std::test]
#[serial_test::serial]
async fn test_gemm_cuda() {
    let backend = cuda_backend();
    test_gemm_generic(&backend).await;
}

// =============================================================================
// test_max_pool_2d
// =============================================================================

/// Test MaxPool2d operation.
async fn test_max_pool_2d_generic(backend: &Arc<GpuBackend>) {
    let ops = Arc::new(LlmOps::new(backend).unwrap());
    let mut cache = TensorCache::default();
    let mut shapes = TensorLayoutBuffers::new(backend);

    let mut ctxt = create_context(backend, &ops, &mut cache, &mut shapes);

    // Create input tensor [N=1, C=1, H=4, W=4] (NCHW format)
    let data = vec![
        1.0f32, 2.0, 3.0, 4.0, // Row 0
        5.0, 6.0, 7.0, 8.0, // Row 1
        9.0, 10.0, 11.0, 12.0, // Row 2
        13.0, 14.0, 15.0, 16.0, // Row 3
    ];
    let input = TensorBuilder::tensor(&[1, 1, 4, 4], BufferUsages::STORAGE)
        .build_init(backend, &data)
        .unwrap();

    // MaxPool with kernel 2x2, stride 2x2
    // Output should be [N=1, C=1, H=2, W=2]
    ctxt.begin_submission();
    let result = ctxt.max_pool_2d(&input, 2, 2, 2, 2, 0, 0).unwrap();
    ctxt.submit();
    backend.synchronize().unwrap();

    let mut output = vec![0.0f32; 4];
    backend
        .slow_read_buffer(result.buffer(), &mut output)
        .await
        .unwrap();

    // Expected: max of each 2x2 block
    // Block (0,0): max(1,2,5,6) = 6
    // Block (0,1): max(3,4,7,8) = 8
    // Block (1,0): max(9,10,13,14) = 14
    // Block (1,1): max(11,12,15,16) = 16
    let expected = [6.0f32, 8.0, 14.0, 16.0];
    for (a, b) in output.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 0.01, "MaxPool mismatch: {} vs {}", a, b);
    }
    println!("MaxPool2d test passed: {:?}", output);
}

#[async_std::test]
#[serial_test::serial]
async fn test_max_pool_2d_webgpu() {
    let backend = create_webgpu_backend().await;
    test_max_pool_2d_generic(&backend).await;
}

#[cfg(feature = "cpu")]
#[async_std::test]
async fn test_max_pool_2d_cpu() {
    let backend = cpu_backend();
    test_max_pool_2d_generic(&backend).await;
}

#[cfg(feature = "cuda")]
#[async_std::test]
#[serial_test::serial]
async fn test_max_pool_2d_cuda() {
    let backend = cuda_backend();
    test_max_pool_2d_generic(&backend).await;
}

// =============================================================================
// test_avg_pool_2d
// =============================================================================

/// Test AvgPool2d operation.
async fn test_avg_pool_2d_generic(backend: &Arc<GpuBackend>) {
    let ops = Arc::new(LlmOps::new(backend).unwrap());
    let mut cache = TensorCache::default();
    let mut shapes = TensorLayoutBuffers::new(backend);

    let mut ctxt = create_context(backend, &ops, &mut cache, &mut shapes);

    // Create input tensor [N=1, C=1, H=4, W=4] (NCHW format)
    let data = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let input = TensorBuilder::tensor(&[1, 1, 4, 4], BufferUsages::STORAGE)
        .build_init(backend, &data)
        .unwrap();

    // AvgPool with kernel 2x2, stride 2x2
    ctxt.begin_submission();
    let result = ctxt.avg_pool_2d(&input, 2, 2, 2, 2, 0, 0, false).unwrap();
    ctxt.submit();
    backend.synchronize().unwrap();

    let mut output = vec![0.0f32; 4];
    backend
        .slow_read_buffer(result.buffer(), &mut output)
        .await
        .unwrap();

    // Expected: avg of each 2x2 block
    // Block (0,0): avg(1,2,5,6) = 3.5
    // Block (0,1): avg(3,4,7,8) = 5.5
    // Block (1,0): avg(9,10,13,14) = 11.5
    // Block (1,1): avg(11,12,15,16) = 13.5
    let expected = [3.5f32, 5.5, 11.5, 13.5];
    for (a, b) in output.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 0.01, "AvgPool mismatch: {} vs {}", a, b);
    }
    println!("AvgPool2d test passed: {:?}", output);
}

#[async_std::test]
#[serial_test::serial]
async fn test_avg_pool_2d_webgpu() {
    let backend = create_webgpu_backend().await;
    test_avg_pool_2d_generic(&backend).await;
}

#[cfg(feature = "cpu")]
#[async_std::test]
async fn test_avg_pool_2d_cpu() {
    let backend = cpu_backend();
    test_avg_pool_2d_generic(&backend).await;
}

#[cfg(feature = "cuda")]
#[async_std::test]
#[serial_test::serial]
async fn test_avg_pool_2d_cuda() {
    let backend = cuda_backend();
    test_avg_pool_2d_generic(&backend).await;
}

// =============================================================================
// test_global_avg_pool_2d
// =============================================================================

/// Test GlobalAvgPool2d operation.
async fn test_global_avg_pool_2d_generic(backend: &Arc<GpuBackend>) {
    let ops = Arc::new(LlmOps::new(backend).unwrap());
    let mut cache = TensorCache::default();
    let mut shapes = TensorLayoutBuffers::new(backend);

    let mut ctxt = create_context(backend, &ops, &mut cache, &mut shapes);

    // Create input tensor [N=1, C=2, H=3, W=3] (NCHW format)
    // Channel 0: all 1s, Channel 1: all 2s
    let data = vec![
        1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // Channel 0
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, // Channel 1
    ];
    let input = TensorBuilder::tensor(&[1, 2, 3, 3], BufferUsages::STORAGE)
        .build_init(backend, &data)
        .unwrap();

    ctxt.begin_submission();
    let result = ctxt.global_avg_pool_2d(&input).unwrap();
    ctxt.submit();
    backend.synchronize().unwrap();

    let mut output = vec![0.0f32; 2];
    backend
        .slow_read_buffer(result.buffer(), &mut output)
        .await
        .unwrap();

    // Expected: global avg of each channel
    // Channel 0: avg(1,1,1,...) = 1.0
    // Channel 1: avg(2,2,2,...) = 2.0
    let expected = [1.0f32, 2.0];
    for (a, b) in output.iter().zip(expected.iter()) {
        assert!(
            (a - b).abs() < 0.01,
            "GlobalAvgPool mismatch: {} vs {}",
            a,
            b
        );
    }
    println!("GlobalAvgPool2d test passed: {:?}", output);
}

#[async_std::test]
#[serial_test::serial]
async fn test_global_avg_pool_2d_webgpu() {
    let backend = create_webgpu_backend().await;
    test_global_avg_pool_2d_generic(&backend).await;
}

#[cfg(feature = "cpu")]
#[async_std::test]
async fn test_global_avg_pool_2d_cpu() {
    let backend = cpu_backend();
    test_global_avg_pool_2d_generic(&backend).await;
}

#[cfg(feature = "cuda")]
#[async_std::test]
#[serial_test::serial]
async fn test_global_avg_pool_2d_cuda() {
    let backend = cuda_backend();
    test_global_avg_pool_2d_generic(&backend).await;
}

// =============================================================================
// test_conv2d_nchw
// =============================================================================

/// Test Conv2d NCHW operation.
async fn test_conv2d_nchw_generic(backend: &Arc<GpuBackend>) {
    let ops = Arc::new(LlmOps::new(backend).unwrap());
    let mut cache = TensorCache::default();
    let mut shapes = TensorLayoutBuffers::new(backend);

    let mut ctxt = create_context(backend, &ops, &mut cache, &mut shapes);

    // Create input tensor [N=1, C=1, H=4, W=4] (NCHW format)
    let input_data = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let input = TensorBuilder::tensor(&[1, 1, 4, 4], BufferUsages::STORAGE)
        .build_init(backend, &input_data)
        .unwrap();

    // Create weight tensor [C_out=1, C_in=1, K_H=2, K_W=2]
    // A simple 2x2 kernel that sums all elements
    let weight_data = vec![1.0f32, 1.0, 1.0, 1.0];
    let weight = TensorBuilder::tensor(&[1, 1, 2, 2], BufferUsages::STORAGE)
        .build_init(backend, &weight_data)
        .unwrap();

    // Conv2d with stride 1, no padding, groups=1
    // Output should be [N=1, C=1, H=3, W=3]
    ctxt.begin_submission();
    let result = ctxt
        .conv_2d_nchw(&input, &weight, 1, 1, 0, 0, 1, 1, 1)
        .unwrap();
    ctxt.submit();
    backend.synchronize().unwrap();

    let mut output = vec![0.0f32; 9];
    backend
        .slow_read_buffer(result.buffer(), &mut output)
        .await
        .unwrap();

    // Expected: sum of each 2x2 window
    // [1+2+5+6, 2+3+6+7, 3+4+7+8] = [14, 18, 22]
    // [5+6+9+10, 6+7+10+11, 7+8+11+12] = [30, 34, 38]
    // [9+10+13+14, 10+11+14+15, 11+12+15+16] = [46, 50, 54]
    let expected = [14.0f32, 18.0, 22.0, 30.0, 34.0, 38.0, 46.0, 50.0, 54.0];
    for (a, b) in output.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 0.01, "Conv2d mismatch: {} vs {}", a, b);
    }
    println!("Conv2d NCHW test passed: {:?}", output);
}

#[async_std::test]
#[serial_test::serial]
async fn test_conv2d_nchw_webgpu() {
    let backend = create_webgpu_backend().await;
    test_conv2d_nchw_generic(&backend).await;
}

#[cfg(feature = "cpu")]
#[async_std::test]
async fn test_conv2d_nchw_cpu() {
    let backend = cpu_backend();
    test_conv2d_nchw_generic(&backend).await;
}

#[cfg(feature = "cuda")]
#[async_std::test]
#[serial_test::serial]
async fn test_conv2d_nchw_cuda() {
    let backend = cuda_backend();
    test_conv2d_nchw_generic(&backend).await;
}
