# Chimera — Multi-Model Agent Orchestrator

## What This Is

Chimera is a Rust-based agent orchestration tool that unifies multiple AI providers
(Grok/xAI, OpenAI, Anthropic, local models via Ollama) behind a single tool-calling
agent loop. Inspired by patterns from OpenAI Codex CLI and claw-code-parity.

## Architecture

- `chimera-providers` — Provider trait + implementations (Grok, OpenAI, Anthropic, Ollama)
- `chimera-tools` — Tool registry, dispatch, and built-in tools (bash, file ops, search)
- `chimera-core` — Agent loop, session management, configuration
- `chimera-cli` — Interactive CLI binary

## Working Rules

- Grok (xAI) is the default provider. Design provider-agnostic, test with Grok first.
- All providers use OpenAI-compatible chat completions wire format where possible.
- Tool definitions use JSON Schema. Tools are registered declaratively.
- The agent loop is synchronous per-turn: send request → stream response → execute tools → loop.
- Prefer simple, flat code over deep abstractions. Add complexity only when needed.
- No unsafe code. Use `anyhow` for application errors, `thiserror` for library errors.
