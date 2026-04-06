use khal_builder::KhalBuilder;

fn main() {
    let shader_crate = "../inferi-shaders";
    let output_dir = "shaders-spirv";

    KhalBuilder::new(shader_crate, true)
        // Mandatory on the web
        .feature("unsafe_remove_boundchecks")
        .build(output_dir);
}
