# inferi-chat

Chat interface for **inferi**.

## Run natively (GUI)

```bash
dx run --release --features desktop
```

## Run natively (CLI)

```bash
cargo run --release --features desktop -- --headless --inspect '/path/to/model.gguf'
```

The CLI does not support Segment Anything.

## Run in the browser

```bash
dx run --release --features web
```
