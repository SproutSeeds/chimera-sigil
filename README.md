# Chimera Sigil

<p align="center">
  <img src="assets/chimera-cover.gif" alt="Chimera Sigil animated cover" width="420" />
</p>

<p align="center">
  Multi-model terminal harness for fast, local-first iteration.
</p>

Chimera Sigil gives you one terminal workflow across Grok, OpenAI, Anthropic, and local Ollama models. It is built for short feedback loops, visible tool use, and fast experimentation without depending on CI spend for every release artifact.

## Mission

Ship a practical multi-model harness that feels fast enough for daily use, clear enough to inspect, and flexible enough to evolve quickly.

## What It Does

- Runs a terminal agent with built-in file, search, edit, and shell tools
- Supports primary models, collaborator models, and fallback chains
- Persists sessions so single-prompt runs can resume later
- Supports approval modes for safer noninteractive automation
- Builds release artifacts locally on macOS, Windows, and Linux/WSL

## Quick Start

Install from npm:

```bash
npm install -g chimera-sigil
chimera --help
```

Run from source:

```bash
cargo run -p chimera-sigil-cli -- --model grok-3
```

Single prompt mode:

```bash
cargo run -p chimera-sigil-cli -- --model sonnet --prompt "Summarize this codebase"
```

Collaborative mode:

```bash
cargo run -p chimera-sigil-cli -- --model grok-3 --collab sonnet,gpt-4o
```

Fallback mode:

```bash
cargo run -p chimera-sigil-cli -- --model grok-3 --fallback gpt-4o,sonnet
```

## Approval Modes

- `--approval-mode prompt`: ask before writes and command execution
- `--approval-mode approve`: allow workspace writes, deny shell execution
- `--approval-mode full`: allow all tools
- `--auto-approve`: compatibility shorthand for `--approval-mode full`

## Environment

Set whichever providers you want to use:

```bash
export XAI_API_KEY=...
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export OLLAMA_BASE_URL=http://localhost:11434/v1
```

No provider keys are committed to this repository. Local machine and release credentials live outside the repo.

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
