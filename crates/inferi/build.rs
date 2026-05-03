use khal_builder::KhalBuilder;

fn main() {
    let output_dir = "shaders-spirv";

    KhalBuilder::from_dependency("inferi-shaders", true)
        // Mandatory on the web
        .feature("unsafe_remove_boundchecks")
        .build(output_dir);
}
