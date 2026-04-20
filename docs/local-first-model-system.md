# Local-First Model System

Chimera should be useful before a user buys credits, configures a cloud key, or owns a high-end GPU. The first path is local, free, and Ollama-compatible. Paid APIs remain available as explicit opt-ins, not as the default route.

## Design Goals

- Start free: `chimera` uses the `local` profile, currently `qwen3:4b`.
- Fit budget laptops: default context budgets stay conservative to avoid memory pressure.
- Keep escape hatches: any Ollama catalog name can be passed directly with `--model`.
- Scale gradually: the same CLI supports tiny CPU models, 7B laptop coding models, and larger local GPU models.
- Make cloud obvious: Grok, OpenAI, and Anthropic aliases still work, but they are no longer the implicit fallback.

## Local Profiles

| User tier | Profile | Model | Notes |
| --- | --- | --- | --- |
| Tiny CPU / low RAM | `local-tiny` | `llama3.2:1b` | Fastest safety net for short turns, routing, summaries, and simple edits. |
| Everyday device | `local-edge` | `gemma3n:e2b` | Good fit for small laptops and mobile-class devices. |
| Budget laptop default | `local`, `local-small`, `local-laptop` | `qwen3:4b` | Default free profile. Small enough to try first, strong enough for real work. |
| Lightweight coding | `local-coder-small` | `qwen2.5-coder:3b` | Code-specific model for constrained machines. |
| Coding laptop | `local-coder`, `local-code` | `qwen2.5-coder:7b` | Better code edits and review on 16GB+ systems. |
| Stronger general local | `local-balanced` | `qwen3:8b` | Use when memory allows and latency is acceptable. |
| Midrange GPU generalist | `local-12gb`, `local-16gb`, `local-heavy` | `qwen3:14b` | Better local reasoning on 12GB/16GB GPUs with modest context. |
| Midrange GPU coding | `local-coder-12gb`, `local-coder-16gb`, `local-coder-heavy` | `qwen2.5-coder:14b` | Code-oriented profile for 12GB/16GB GPUs. |
| Reasoning pass | `local-reasoning` | `deepseek-r1:8b` | Slower independent reasoning for hard decisions. |
| 24GB GPU generalist | `local-24gb`, `local-4090`, `local-workstation`, `local-gpu` | `qwen3:30b` | Bigger local-GPU profile for cards like an RTX 4090. |
| 24GB GPU coding | `local-coder-24gb`, `local-coder-4090`, `local-coder-gpu` | `qwen2.5-coder:32b` | 24GB coding profile with a conservative context budget. |

## Recommended Combinations

Minimum free path:

```bash
chimera --local-doctor
chimera --model local
```

Lowest-memory fallback chain:

```bash
chimera --model local --fallback local-tiny
```

Budget coding laptop:

```bash
chimera --model local-coder-small --fallback local,local-tiny
```

16GB+ coding laptop:

```bash
chimera --model local-coder --collab local-tiny
```

12GB/16GB GPU coding:

```bash
chimera --model local-coder-16gb --fallback local-coder,local
```

Stronger local generalist:

```bash
chimera --model local-balanced --collab local-coder-small
```

RTX 4090 or other 24GB GPU generalist:

```bash
chimera --model local-4090 --fallback local-16gb,local-balanced
```

RTX 4090 or other 24GB GPU coding:

```bash
chimera --model local-coder-4090 --collab local-coder-small
```

On small machines, prefer fallback over collaboration. Fallback only runs when the primary provider fails; collaboration asks multiple models each turn and can be slow or memory-heavy when all models are local.

## Setup

Pull the default model:

```bash
ollama pull qwen3:4b
```

Pull the coding profile:

```bash
ollama pull qwen2.5-coder:7b
```

If the default model is too slow or fails to load:

```bash
ollama pull llama3.2:1b
chimera --model local-tiny
```

## Implementation Notes

- `resolve_model` maps local aliases to Ollama model IDs.
- Unknown model names now route to Ollama, which makes no-key catalog names like `gemma3n` or `qwen3` work directly.
- Cloud prefixes still route explicitly: `grok*` to xAI, `gpt*`/`o*`/`codex*` to OpenAI, and `claude*` to Anthropic.
- Context windows for local profiles are working budgets, not published maximums. The goal is stability on consumer hardware.
- Ollama uses the OpenAI-compatible `/v1/chat/completions` API, including streaming and tools, so the existing provider can serve local models.

## Source Pointers

- Ollama OpenAI compatibility: https://docs.ollama.com/api/openai-compatibility
- Qwen3 on Ollama: https://ollama.com/library/qwen3
- Qwen2.5 Coder on Ollama: https://ollama.com/library/qwen2.5-coder
- Gemma 3n on Ollama: https://ollama.com/library/gemma3n
- DeepSeek-R1 on Ollama: https://ollama.com/library/deepseek-r1
