//! ONNX graph representation and topological sorting.

use crate::onnx::error::OnnxError;
use onnx_protobuf::{AttributeProto, GraphProto, NodeProto, TensorProto, ValueInfoProto};
use std::collections::{HashMap, HashSet, VecDeque};

/// A processed ONNX graph ready for execution planning.
#[derive(Debug)]
pub struct OnnxGraph {
    /// Nodes in topologically sorted order (execution order).
    pub nodes: Vec<OnnxNode>,
    /// Input tensor names and their expected shapes (None for dynamic dims).
    pub inputs: Vec<GraphInput>,
    /// Output tensor names.
    pub outputs: Vec<String>,
    /// Initializer tensors (model weights).
    pub initializers: HashMap<String, TensorProto>,
}

/// A single node in the graph.
#[derive(Debug, Clone)]
pub struct OnnxNode {
    /// Node name (for error reporting).
    pub name: String,
    /// Operation type (e.g., "MatMul", "Add", "Relu").
    pub op_type: String,
    /// Input tensor names.
    pub inputs: Vec<String>,
    /// Output tensor names.
    pub outputs: Vec<String>,
    /// Node attributes.
    pub attributes: HashMap<String, AttributeValue>,
}

/// A graph input specification.
#[derive(Debug, Clone)]
pub struct GraphInput {
    /// Input tensor name.
    pub name: String,
    /// Expected shape (None values indicate dynamic dimensions).
    pub shape: Option<Vec<Option<u32>>>,
}

/// Parsed attribute values.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum AttributeValue {
    Float(f32),
    Int(i64),
    String(String),
    Floats(Vec<f32>),
    Ints(Vec<i64>),
    Tensor(TensorProto),
}

impl OnnxGraph {
    /// Build an OnnxGraph from a GraphProto.
    pub fn from_proto(graph: &GraphProto) -> Result<Self, OnnxError> {
        // Collect initializers (model weights)
        let initializers: HashMap<String, TensorProto> = graph
            .initializer
            .iter()
            .map(|t| (t.name.clone(), t.clone()))
            .collect();

        // Collect input names (excluding initializers, which are also listed as inputs)
        let initializer_names: HashSet<&str> = initializers.keys().map(|s| s.as_str()).collect();
        let inputs: Vec<GraphInput> = graph
            .input
            .iter()
            .filter(|i| !initializer_names.contains(i.name.as_str()))
            .map(parse_value_info)
            .collect();

        // Collect output names
        let outputs: Vec<String> = graph.output.iter().map(|o| o.name.clone()).collect();

        // Parse and topologically sort nodes
        let nodes = topological_sort(&graph.node)?;

        Ok(Self {
            nodes,
            inputs,
            outputs,
            initializers,
        })
    }

    /// Get the set of all tensor names that are graph inputs or initializers.
    pub fn available_inputs(&self) -> HashSet<&str> {
        let mut available: HashSet<&str> = HashSet::new();
        for input in &self.inputs {
            available.insert(&input.name);
        }
        for name in self.initializers.keys() {
            available.insert(name);
        }
        available
    }
}

/// Parse a ValueInfoProto to extract input specification.
fn parse_value_info(info: &ValueInfoProto) -> GraphInput {
    let shape = info.type_.0.as_ref().and_then(|t| {
        t.value.as_ref().and_then(|v| match v {
            onnx_protobuf::type_proto::Value::TensorType(tensor_type) => {
                tensor_type.shape.0.as_ref().map(|shape| {
                    shape
                        .dim
                        .iter()
                        .map(|d| {
                            if d.has_dim_value() {
                                Some(d.dim_value() as u32)
                            } else {
                                None // Dynamic dimension
                            }
                        })
                        .collect()
                })
            }
            _ => None,
        })
    });

    GraphInput {
        name: info.name.clone(),
        shape,
    }
}

/// Parse a NodeProto into an OnnxNode.
fn parse_node(proto: &NodeProto) -> OnnxNode {
    let attributes: HashMap<String, AttributeValue> = proto
        .attribute
        .iter()
        .filter_map(|a| parse_attribute(a).map(|v| (a.name.clone(), v)))
        .collect();

    OnnxNode {
        name: if proto.name.is_empty() {
            proto.output.first().cloned().unwrap_or_default()
        } else {
            proto.name.clone()
        },
        op_type: proto.op_type.clone(),
        inputs: proto.input.clone(),
        outputs: proto.output.clone(),
        attributes,
    }
}

/// Parse an AttributeProto into an AttributeValue.
fn parse_attribute(attr: &AttributeProto) -> Option<AttributeValue> {
    use onnx_protobuf::attribute_proto::AttributeType;
    match attr.type_.enum_value() {
        Ok(AttributeType::FLOAT) => Some(AttributeValue::Float(attr.f)),
        Ok(AttributeType::INT) => Some(AttributeValue::Int(attr.i)),
        Ok(AttributeType::STRING) => Some(AttributeValue::String(
            String::from_utf8_lossy(&attr.s).into_owned(),
        )),
        Ok(AttributeType::FLOATS) => Some(AttributeValue::Floats(attr.floats.clone())),
        Ok(AttributeType::INTS) => Some(AttributeValue::Ints(attr.ints.clone())),
        Ok(AttributeType::TENSOR) => attr
            .t
            .0
            .as_ref()
            .map(|t| AttributeValue::Tensor((**t).clone())),
        _ => None,
    }
}

/// Perform topological sort on graph nodes using Kahn's algorithm.
fn topological_sort(nodes: &[NodeProto]) -> Result<Vec<OnnxNode>, OnnxError> {
    // Build adjacency information
    // Map from tensor name to the node that produces it
    let mut tensor_producer: HashMap<&str, usize> = HashMap::new();
    for (idx, node) in nodes.iter().enumerate() {
        for output in &node.output {
            if !output.is_empty() {
                tensor_producer.insert(output.as_str(), idx);
            }
        }
    }

    // Calculate in-degree (number of node dependencies) for each node
    let mut in_degree: Vec<usize> = vec![0; nodes.len()];
    let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];

    for (idx, node) in nodes.iter().enumerate() {
        let mut deps = HashSet::new();
        for input in &node.input {
            if let Some(&producer_idx) = tensor_producer.get(input.as_str()) {
                if producer_idx != idx && deps.insert(producer_idx) {
                    in_degree[idx] += 1;
                    dependents[producer_idx].push(idx);
                }
            }
        }
    }

    // Kahn's algorithm
    let mut queue: VecDeque<usize> = VecDeque::new();
    for (idx, &degree) in in_degree.iter().enumerate() {
        if degree == 0 {
            queue.push_back(idx);
        }
    }

    let mut sorted: Vec<OnnxNode> = Vec::with_capacity(nodes.len());
    let mut visited = 0;

    while let Some(idx) = queue.pop_front() {
        sorted.push(parse_node(&nodes[idx]));
        visited += 1;

        for &dependent in &dependents[idx] {
            in_degree[dependent] -= 1;
            if in_degree[dependent] == 0 {
                queue.push_back(dependent);
            }
        }
    }

    if visited != nodes.len() {
        // Find a node involved in the cycle for error reporting
        let cycle_node = nodes
            .iter()
            .enumerate()
            .find(|(idx, _)| in_degree[*idx] > 0)
            .map(|(_, n)| n.name.clone())
            .unwrap_or_else(|| "unknown".to_string());
        return Err(OnnxError::CyclicGraph(cycle_node));
    }

    Ok(sorted)
}

impl OnnxNode {
    /// Get a float attribute, returning an error if missing or wrong type.
    pub fn get_float_attr(&self, name: &str) -> Result<f32, OnnxError> {
        match self.attributes.get(name) {
            Some(AttributeValue::Float(v)) => Ok(*v),
            Some(_) => Err(OnnxError::InvalidAttribute {
                attr: name.to_string(),
                node: self.name.clone(),
                reason: "expected float".to_string(),
            }),
            None => Err(OnnxError::InvalidAttribute {
                attr: name.to_string(),
                node: self.name.clone(),
                reason: "attribute not found".to_string(),
            }),
        }
    }

    /// Get a float attribute with a default value.
    pub fn get_float_attr_or(&self, name: &str, default: f32) -> f32 {
        match self.attributes.get(name) {
            Some(AttributeValue::Float(v)) => *v,
            _ => default,
        }
    }

    /// Get an int attribute, returning an error if missing or wrong type.
    pub fn get_int_attr(&self, name: &str) -> Result<i64, OnnxError> {
        match self.attributes.get(name) {
            Some(AttributeValue::Int(v)) => Ok(*v),
            Some(_) => Err(OnnxError::InvalidAttribute {
                attr: name.to_string(),
                node: self.name.clone(),
                reason: "expected int".to_string(),
            }),
            None => Err(OnnxError::InvalidAttribute {
                attr: name.to_string(),
                node: self.name.clone(),
                reason: "attribute not found".to_string(),
            }),
        }
    }

    /// Get an int attribute with a default value.
    pub fn get_int_attr_or(&self, name: &str, default: i64) -> i64 {
        match self.attributes.get(name) {
            Some(AttributeValue::Int(v)) => *v,
            _ => default,
        }
    }

    /// Get an ints attribute, returning an error if missing or wrong type.
    pub fn get_ints_attr(&self, name: &str) -> Result<&[i64], OnnxError> {
        match self.attributes.get(name) {
            Some(AttributeValue::Ints(v)) => Ok(v),
            Some(_) => Err(OnnxError::InvalidAttribute {
                attr: name.to_string(),
                node: self.name.clone(),
                reason: "expected ints".to_string(),
            }),
            None => Err(OnnxError::InvalidAttribute {
                attr: name.to_string(),
                node: self.name.clone(),
                reason: "attribute not found".to_string(),
            }),
        }
    }

    /// Get an ints attribute with a default value.
    pub fn get_ints_attr_or<'a>(&'a self, name: &str, default: &'a [i64]) -> &'a [i64] {
        match self.attributes.get(name) {
            Some(AttributeValue::Ints(v)) => v,
            _ => default,
        }
    }

    /// Get a string attribute, returning an error if missing or wrong type.
    pub fn get_string_attr(&self, name: &str) -> Result<&str, OnnxError> {
        match self.attributes.get(name) {
            Some(AttributeValue::String(v)) => Ok(v),
            Some(_) => Err(OnnxError::InvalidAttribute {
                attr: name.to_string(),
                node: self.name.clone(),
                reason: "expected string".to_string(),
            }),
            None => Err(OnnxError::InvalidAttribute {
                attr: name.to_string(),
                node: self.name.clone(),
                reason: "attribute not found".to_string(),
            }),
        }
    }

    /// Get a string attribute, returning default if missing.
    pub fn get_string_attr_or<'a>(&'a self, name: &str, default: &'a str) -> &'a str {
        self.get_string_attr(name).unwrap_or(default)
    }

    /// Get a tensor attribute, returning an error if missing or wrong type.
    pub fn get_tensor_attr(&self, name: &str) -> Result<&TensorProto, OnnxError> {
        match self.attributes.get(name) {
            Some(AttributeValue::Tensor(t)) => Ok(t),
            Some(_) => Err(OnnxError::InvalidAttribute {
                attr: name.to_string(),
                node: self.name.clone(),
                reason: "expected tensor".to_string(),
            }),
            None => Err(OnnxError::InvalidAttribute {
                attr: name.to_string(),
                node: self.name.clone(),
                reason: "attribute not found".to_string(),
            }),
        }
    }
}
