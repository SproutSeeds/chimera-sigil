# Contributing

Thanks for helping build Chimera Sigil.

## Philosophy

- Prefer small, composable changes over broad rewrites
- Optimize for fast feedback and clear behavior
- Keep the tool loop inspectable and easy to reason about

## Setup

Required:

- Rust toolchain
- Node.js 18+

Optional provider environment:

```bash
export XAI_API_KEY=...
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export OLLAMA_BASE_URL=http://localhost:11434/v1
```

## Local Workflow

Run before opening a PR:

```bash
cargo fmt
cargo test
cargo run -p chimera-sigil-cli -- --help
```

If you touch packaging or release logic, also run:

```bash
node scripts/release-local.mjs --doctor
```

## Pull Requests

- Keep PRs focused on one change area
- Add or update tests when behavior changes
- Update `README.md` when user-facing CLI behavior changes
- Call out tradeoffs, follow-ups, and known gaps clearly

## Release Workflow

Chimera Sigil is designed for local-first releases.

- macOS artifacts can be built locally
- Windows and Linux artifacts can be built through the configured Windows + WSL host
- `node scripts/release-local.mjs` builds the full artifact set when the remote builder is ready

## Mission Fit

The strongest contributions improve at least one of these:

- speed of iteration
- clarity of model and tool behavior
- reliability of local-first release infrastructure
- practical multi-model collaboration
