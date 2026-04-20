# Chimera Sigil

<p align="center">
  <img src="assets/chimera-cover.gif" alt="Chimera Sigil animated cover" width="420" />
</p>

<p align="center">
  Free local-first terminal agent for Ollama, local GPUs, and optional cloud models.
</p>

Chimera Sigil gives you one terminal workflow that starts with local Ollama models and can still reach Grok, OpenAI, or Anthropic when you explicitly choose them. It is built for short feedback loops, visible tool use, and fast experimentation without requiring API spend.

## Mission

Ship a practical local-first multi-model harness that lets people get real work done for free on the hardware they already have, then scales up to bigger local GPUs or paid APIs only when they choose that path.

## What It Does

- Runs a terminal agent with built-in file, search, edit, and shell tools
- Supports local model profiles, collaborator models, and fallback chains
- Persists sessions so single-prompt runs can resume later
- Supports approval modes for safer noninteractive automation
- Builds release artifacts locally on macOS, Windows, and Linux/WSL

## Quick Start

Install from npm:

```bash
npm install -g chimera-sigil
chimera --local-doctor
ollama pull qwen3:4b
chimera --help
```

Install and start Ollama first if it is not already running.

Run from source:

```bash
cargo run -p chimera-sigil-cli
```

Single prompt mode:

```bash
cargo run -p chimera-sigil-cli -- --prompt "Summarize this codebase"
```

Local coding profile:

```bash
cargo run -p chimera-sigil-cli -- --model local-coder --prompt "Find the risky parts of this patch"
```

Collaborative local mode:

```bash
cargo run -p chimera-sigil-cli -- --model local-small --collab local-tiny,local-coder
```

Fallback mode:

```bash
cargo run -p chimera-sigil-cli -- --model local-coder --fallback local-small,local-tiny
```

## Local Model Profiles

The default CLI model is `local`, which resolves to `qwen3:4b` through Ollama. These aliases are intentionally conservative so a person with a newer budget laptop can start useful work without buying API credits.

| Profile | Ollama model | Best fit |
| --- | --- | --- |
| `local-tiny` | `llama3.2:1b` | Lowest memory, quick notes, routing, simple edits |
| `local-edge` | `gemma3n:e2b` | Everyday-device profile for laptops and small machines |
| `local`, `local-small`, `local-laptop` | `qwen3:4b` | Default free laptop profile |
| `local-coder-small` | `qwen2.5-coder:3b` | Lightweight coding assistant |
| `local-coder`, `local-code` | `qwen2.5-coder:7b` | Better code review and edits on 16GB+ machines |
| `local-balanced` | `qwen3:8b` | Stronger general work when memory allows |
| `local-12gb`, `local-16gb`, `local-heavy` | `qwen3:14b` | Midrange NVIDIA GPUs and larger-memory laptops |
| `local-coder-12gb`, `local-coder-16gb`, `local-coder-heavy` | `qwen2.5-coder:14b` | Midrange GPU coding profile |
| `local-reasoning` | `deepseek-r1:8b` | Slower reasoning pass for hard problems |
| `local-24gb`, `local-4090`, `local-workstation`, `local-gpu` | `qwen3:30b` | 24GB GPU generalist profile |
| `local-coder-24gb`, `local-coder-4090`, `local-coder-gpu` | `qwen2.5-coder:32b` | 24GB GPU coding profile |

Any Ollama model name also works directly:

```bash
chimera --model gemma3n
chimera --model qwen3:8b
chimera --model qwen2.5-coder:14b
```

For the full hardware matrix and benchmarking plan, see [docs/hardware-optimization-process.md](docs/hardware-optimization-process.md).

Ask Chimera to recommend a local profile for the current machine:

```bash
chimera --local-doctor
chimera --local-doctor --json
```

Benchmark local profiles with Ollama timing metrics:

```bash
chimera --local-benchmark
chimera --local-benchmark --benchmark-models local-4090,local-coder-4090,local-coder-16gb
```

## Approval Modes

- `--approval-mode prompt`: ask before writes and command execution
- `--approval-mode approve`: allow workspace writes, deny shell execution
- `--approval-mode full`: allow all tools
- `--auto-approve`: compatibility shorthand for `--approval-mode full`

## Environment

Local Ollama needs no API key. Cloud providers are optional:

```bash
export OLLAMA_BASE_URL=http://localhost:11434/v1
export XAI_API_KEY=...
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
```

No provider keys are committed to this repository. Local machine, Ollama, and release credentials live outside the repo.

## Fast Iteration

- Prompt mode auto-saves sessions for later resume
- Release builds can be produced locally instead of relying on GitHub Actions
- The shipped CLI command stays short: `chimera`

## Contributing

Contributions are welcome. Keep changes small, tested, and easy to review.

Start here:

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [SECURITY.md](SECURITY.md)

## Project Layout

- `crates/chimera-core`: agent loop, config, sessions
- `crates/chimera-providers`: model providers and streaming
- `crates/chimera-tools`: built-in tools and permissions
- `crates/chimera-cli`: CLI entrypoint
- `scripts/`: packaging and local-first release tooling

## Release Notes

The npm package name is `chimera-sigil`. The installed command is `chimera`.

Local-first release tooling currently supports:

- `aarch64-apple-darwin`
- `x86_64-apple-darwin`
- `x86_64-pc-windows-msvc`
- `x86_64-unknown-linux-gnu`

For contributor setup and release workflow details, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Support

Everything here is released for public use. If Chimera saved you time or you want to keep the work moving, you can [support public FRG releases](https://frg.earth/support?utm_source=readme&utm_medium=repo&utm_campaign=public_work_support&package=chimera-sigil).
