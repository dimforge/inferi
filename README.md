# inferi − on-device GPU inference on every platform

<p align="center">
  <img src="./crates/inferi-chat/assets/inferi-logo.png" height="200px">
</p>
<p align="center">
    <a href="https://discord.gg/vt9DJSW">
        <img src="https://img.shields.io/discord/507548572338880513.svg?logo=discord&colorB=7289DA">
    </a>
</p>

-----

**inferi** is a set of Rust libraries for local LLM and vision model inference on the GPU.
Shaders are written in Rust via [rust-gpu](https://github.com/Rust-GPU/rust-gpu), targeting
Vulkan/WebGPU through [wgpu](https://github.com/gfx-rs/wgpu). It runs natively on desktop, mobile,
and in the browser.

> **Warning:** inferi is under active development and may be missing features. Contributions welcome!

## Getting started

Install [cargo-gpu](https://github.com/Rust-GPU/cargo-gpu) (required to compile shaders):

```bash
cargo install cargo-gpu --version 0.10.0-alpha.1
cargo gpu install
```

Run the chat app natively:

```bash
cd crates/inferi-chat
dx run --release --features desktop
```

Or the CLI version:

```bash
cd crates/inferi-chat
cargo run --release --features desktop -- --headless --inspect '/path/to/model.gguf'
```
