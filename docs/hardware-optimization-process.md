# Hardware Optimization Process

Local-first does not mean one model for every computer. Chimera should pick a model arrangement from the user's actual hardware, then prove that arrangement with repeatable benchmarks.

## Hardware Tiers

| Tier | Typical specs | Primary profile | Coding profile | Notes |
| --- | --- | --- | --- | --- |
| Edge / tiny | CPU-only, 8GB RAM, weak or no GPU | `local-tiny` | `local-coder-small` only if memory allows | Keep prompts short and avoid collaboration. |
| Budget laptop | Newer $600-class laptop, 12GB-16GB RAM, integrated GPU or small dGPU | `local` | `local-coder-small` | Default target. Useful work for free, with conservative context. |
| Everyday laptop+ | 16GB-24GB RAM or Apple/AMD unified memory | `local-balanced` | `local-coder` | Good daily driver when `qwen3:4b` is too small. |
| 8GB VRAM GPU | RTX 4060-class, older 2070/3070-class cards | `local-balanced` | `local-coder` | Watch context growth; stay near 8K-16K for larger edits. |
| 12GB VRAM GPU | RTX 3060 12GB, 4070, 4070 SUPER | `local-12gb` | `local-coder-12gb` | Use 14B models when they stay fully on GPU. |
| 16GB VRAM GPU | 4060 Ti 16GB, 4070 Ti SUPER, 4080-class | `local-16gb` | `local-coder-16gb` | Strong local tier for serious coding and review. |
| 24GB VRAM GPU | RTX 3090, RTX 4090 | `local-4090` | `local-coder-4090` | Best consumer local lane; keep 30B/32B contexts conservative. |
| 40GB+ VRAM GPU | A100 40GB, L40S 48GB, A6000 48GB | `local-workstation` plus direct model names | direct 32B/70B tests | Use when testing frontier-adjacent local workflows. |

The RTX 4090 lane should not be forced onto budget laptops. It is a separate quality tier: `qwen3:30b` for general reasoning and `qwen2.5-coder:32b` for code-heavy work.

## Selection Algorithm

Start with the built-in local doctor:

```bash
chimera --local-doctor
chimera --local-doctor --json
```

The doctor is intentionally a recommendation pass, not a benchmark. It fingerprints the machine, suggests an initial primary/coding/fallback set, and names the profiles that should be benchmarked next.

Then run the local benchmark:

```bash
chimera --local-benchmark
chimera --local-benchmark --benchmark-models local-4090,local-coder-4090,local-coder-16gb
```

The benchmark emits JSONL. It uses Ollama's local `/api/generate` endpoint and records wall time, load time, prompt eval time, eval time, token counts, and generated-token throughput for a small set of deterministic prompts.

1. Fingerprint the machine.
   - OS and architecture
   - CPU core count
   - system RAM
   - GPU name
   - GPU VRAM
   - driver/runtime status
   - Ollama availability

2. Pick a candidate set.
   - No GPU or low RAM: `local-tiny`, `local`, `local-coder-small`
   - 8GB GPU: `local-balanced`, `local-coder`
   - 12GB/16GB GPU: `local-16gb`, `local-coder-16gb`, fallback to 7B models
   - 24GB GPU: `local-4090`, `local-coder-4090`, fallback to 14B models
   - 40GB+ GPU: test larger direct Ollama model names in addition to aliases

3. Pull only the candidate models.
   - Pulling everything wastes disk and time.
   - Always include one small fallback model for recovery.

4. Run a smoke prompt.
   - Verify the model loads.
   - Verify streaming works.
   - Verify a tool call can be produced and parsed.

5. Run the benchmark suite.
   - Summarization of a medium file
   - Code search and explanation
   - Patch planning
   - Single-file edit
   - Multi-file edit
   - Tool-call JSON reliability
   - Long-context degradation

6. Record operational metrics.
   - time to first token
   - output tokens per second
   - prompt processing tokens per second
   - peak VRAM
   - peak RAM
   - CPU fallback or GPU residency
   - completion success rate
   - tool-call success rate

7. Choose the arrangement.
   - Primary: highest-quality model that stays acceptably fast and resident.
   - Fallback: smaller model known to load.
   - Collaborators: only add when the machine can keep latency reasonable.

8. Save a local recommendation.
   - Future shape: `.chimera/local-profile.json`
   - Store detected hardware, chosen profiles, benchmark scores, and model pull status.

## 4090 WSL Rig Plan

For the Windows + WSL RTX 4090 machine, the first benchmark lane should be:

```bash
chimera --model local-4090 --fallback local-16gb,local-balanced
chimera --model local-coder-4090 --fallback local-coder-16gb,local-coder
chimera --model local-coder-4090 --collab local-coder-small
```

Acceptance targets:

- The 30B/32B primary model should stay mostly GPU-resident.
- Context should begin conservative, around 8K-16K effective workload, before trying larger contexts.
- If VRAM spills heavily, demote the profile to the 14B lane.
- Collaboration should be optional, not default, until latency is measured.

Useful remote checks:

```bash
nvidia-smi
ollama --version
ollama list
ollama ps
```

## Test Lab Strategy

Use physical machines for consumer GPU truth. VMs are useful for CPU/RAM matrix testing, but GPU performance must be tested with real GPU access or cloud GPU passthrough.

Recommended matrix:

| Target | Purpose |
| --- | --- |
| CPU-only VM, 8GB RAM | lowest viable behavior |
| CPU-only VM, 16GB RAM | budget non-GPU laptop approximation |
| Local budget laptop | real thermal and memory behavior |
| Windows/WSL RTX 4090 | 24GB consumer GPU lane |
| Cloud 16GB GPU | midrange reproducibility lane |
| Cloud 40GB+ GPU | large-model and training experiments |

Brev is a good fit for temporary cloud GPU lanes when we need repeatable A10/L4/A100/L40S-style tests without owning every card. The local 4090 remains the key consumer benchmark because it represents a very strong machine that a serious individual user could actually own.

## Fine-Tuning And Specialized Local Models

The long-term stack should use frontier models to help create local specialists:

- Generate high-quality task traces with frontier models.
- Distill those traces into small local models.
- Fine-tune task specialists for narrow jobs: search planning, patch review, test failure triage, doc summarization, release note drafting, and structured code edits.
- Route easy or repetitive work to local specialists before spending API tokens.
- Escalate only when local confidence is low or the task crosses a complexity threshold.

The goal is not one local model that replaces everything. The goal is a swarm of small, cheap, local specialists that absorb predictable work and reserve frontier models for the genuinely hard frontier-shaped problems.

## Source Pointers

- Ollama OpenAI-compatible endpoint supports chat completions, streaming, and tools: https://docs.ollama.com/api/openai-compatibility
- NVIDIA lists RTX 4090 memory as 24GB: https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/
- Ollama model sizes and contexts: https://ollama.com/library/qwen3, https://ollama.com/library/qwen2.5-coder, https://ollama.com/library/llama3.2, https://ollama.com/library/gemma3n, https://ollama.com/library/deepseek-r1
