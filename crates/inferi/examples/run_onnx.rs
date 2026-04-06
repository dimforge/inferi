//! Example: Run inference on an ONNX model.
//!
//! Usage:
//!   cargo run --example run_onnx --features onnx -- model.onnx
//!
//! This example loads an ONNX model, creates random input tensors matching
//! the model's expected input shapes, runs inference, and prints the output shapes.

use clap::Parser;
use std::path::PathBuf;

#[cfg(feature = "onnx")]
use inferi::context::{LlmContext, LlmOps};
#[cfg(feature = "onnx")]
use inferi::tensor_cache::TensorCache;
#[cfg(feature = "onnx")]
use khal::backend::{Backend, GpuBackend, WebGpu};
#[cfg(feature = "onnx")]
use khal::BufferUsages;
#[cfg(feature = "onnx")]
use std::collections::HashMap;
#[cfg(feature = "onnx")]
use std::sync::Arc;
#[cfg(feature = "onnx")]
use vortx::shapes::TensorLayoutBuffers;
#[cfg(feature = "onnx")]
use vortx::tensor::TensorBuilder;
#[cfg(feature = "onnx")]
use wgpu::{Features, Limits};

#[cfg(feature = "onnx")]
use inferi::onnx::OnnxModel;

#[derive(Parser, Debug)]
#[command(name = "run_onnx")]
#[command(about = "Run inference on an ONNX model")]
struct Args {
    /// Path to the ONNX model file
    model: PathBuf,

    /// Override batch size (default: 1)
    #[arg(short, long, default_value = "1")]
    batch_size: u32,
}

#[cfg(feature = "onnx")]
#[async_std::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("Loading ONNX model from: {}", args.model.display());

    // Load the ONNX model
    let model = OnnxModel::from_file(&args.model)?;

    println!("Model loaded successfully!");
    println!("  IR version: {}", model.ir_version());
    println!("  Opset version: {}", model.opset_version());
    println!("  Producer: {}", model.producer_name());

    let model_inputs = model.inputs()?;
    let model_outputs = model.outputs()?;

    println!("  Inputs:");
    for input in &model_inputs {
        println!("    - {} {:?}", input.name, input.shape);
    }
    println!("  Outputs:");
    for output in &model_outputs {
        println!("    - {}", output);
    }

    println!("\nOperations:");
    model.print_operations()?;

    // Initialize GPU backend
    println!("\nInitializing GPU backend...");
    let webgpu = WebGpu::new(Features::default(), Limits::default()).await?;
    let backend = Arc::new(GpuBackend::WebGpu(webgpu));

    // Build input shapes map (use provided shapes or defaults)
    let mut input_shapes = HashMap::new();
    for input in &model_inputs {
        let shape: Vec<u32> = input
            .shape
            .as_ref()
            .map(|s| s.iter().map(|d| d.unwrap_or(args.batch_size)).collect())
            .unwrap_or_else(|| vec![args.batch_size]);
        println!("Using input shape for '{}': {:?}", input.name, shape);
        input_shapes.insert(input.name.clone(), shape);
    }

    // Compile the model for GPU execution
    println!("\nCompiling model...");
    let compiled = model.compile(&backend, &input_shapes)?;
    println!("Model compiled successfully!");

    // Create context for inference
    let ops = Arc::new(LlmOps::new(&backend)?);
    let mut cache = TensorCache::default();
    let mut shapes = TensorLayoutBuffers::new(&backend);

    let mut ctxt = LlmContext {
        backend: &backend,
        cache: &mut cache,
        shapes: &mut shapes,
        pass: None,
        encoder: None,
        ops: &ops,
    };

    // Create random input tensors
    println!("\nCreating input tensors...");
    let mut input_tensors = Vec::new();

    for (name, shape) in &input_shapes {
        let total_elements: usize = shape.iter().map(|&d| d as usize).product();
        let data: Vec<f32> = (0..total_elements).map(|i| (i as f32) * 0.01).collect();
        let tensor =
            TensorBuilder::tensor(shape, BufferUsages::STORAGE).build_init(&backend, &data)?;
        input_tensors.push((name.clone(), tensor));
    }

    let mut inputs = HashMap::new();
    for (name, tensor) in &input_tensors {
        inputs.insert(name.clone(), tensor);
    }

    // Run inference
    println!("Running inference...");
    ctxt.begin_submission();
    let outputs = compiled.run(&mut ctxt, inputs).await?;
    ctxt.submit();
    backend.synchronize()?;

    // Print output information
    println!("\nOutputs:");
    for (name, tensor) in &outputs {
        println!("  {}: shape {:?}", name, tensor.shape());
    }

    // Optionally read back first few values of each output
    println!("\nOutput values (first 10 elements):");
    for (name, tensor) in &outputs {
        let total_len = tensor.len() as usize;
        let read_len = total_len.min(10);
        let mut values = vec![0.0f32; total_len];
        backend
            .slow_read_buffer(tensor.buffer(), &mut values)
            .await?;
        println!("  {}: {:?}", name, &values[..read_len]);
    }

    println!("\nDone!");
    Ok(())
}

#[cfg(not(feature = "onnx"))]
fn main() {
    eprintln!("Error: This example requires the 'onnx' feature.");
    eprintln!("Run with: cargo run --example run_onnx --features onnx -- model.onnx");
    std::process::exit(1);
}
