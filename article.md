# Announcing RapidFire AI Official Hugging Face TRL Integration

Today is a big milestone for RapidFire AI: we’re officially integrated into Hugging Face’s TRL documentation as a first-class integration. That means TRL users can now discover, install, and run RapidFire AI as the fastest way to compare many fine-tuning/post-training configurations—without changing their workflow.

## Why this matters

Teams don't have the time (or budget) to train one config after another. Our TRL integration lets you launch many TRL configurations concurrently—even on a single GPU—via adaptive, chunk-based scheduling. In internal benchmarks referenced in the TRL page, this delivers ~16–24× higher experimentation throughput than sequential runs, so you reach better models dramatically faster.

![RapidFire AI Architecture](images/rf-usage.png)
*RapidFire AI establishes live three-way communication between your IDE, a metrics dashboard, and a multi-GPU execution backend*

## What you get, out of the box

- **Drop-in TRL wrappers** — Use `RFSFTConfig`, `RFDPOConfig`, and `RFGRPOConfig` as near-zero-code replacements for TRL's SFT/DPO/GRPO configs.

- **Chunk-based concurrent training** — We shard data and cycle configs at chunk boundaries to maximize GPU utilization and enable early, apples-to-apples, comparisons.

- **Interactive Control Ops (IC Ops)** — From the dashboard, Stop, Resume, Clone, and Clone & Warm-Start any run in flight to double-down on winners and pause stragglers—no job restarts required.

![Interactive Control Operations](images/icop-clone.png)
*Clone promising configurations with modified hyperparameters—optionally warm-starting from the parent's weights—all from the live dashboard*

- **Multi-GPU orchestration** — The scheduler auto-distributes configs across available GPUs; you focus on models, not plumbing.

- **MLflow-based dashboard** — Real-time metrics, logs, and IC Ops in one place as soon as you start your experiment.

## How it works

RapidFire AI slices your dataset into "chunks" and rotates configurations through the GPU at chunk boundaries. You get incremental signal on all configs quickly, while automatic checkpointing keeps training stable. Then, use IC Ops to adapt mid-flight—stop low-performers early and clone promising ones with tweaked hyperparameters (optionally warm-starting from the parent's weights).

![GPU Scheduling Comparison](images/gantt-2gpu.png)
*Sequential vs. Task Parallel vs. RapidFire AI: Our adaptive scheduler maximizes GPU utilization across multiple configs and GPUs. The bottom row shows IC Ops in action—stopping, cloning, and modifying runs mid-flight.*

## Getting Started

Install RapidFire AI and get running in under a minute:

```bash
pip install rapidfireai

# Authenticate with Hugging Face
huggingface-cli login --token YOUR_TOKEN

# Workaround for current issue
pip uninstall -y hf-xet

# Initialize and start RapidFire AI
rapidfireai init
rapidfireai start
```

The dashboard launches at `http://localhost:3000` where you can monitor and control all your experiments.

## Supported TRL trainers

- SFT with RFSFTConfig
- DPO with RFDPOConfig
- GRPO with RFGRPOConfig

These are designed as drop-in replacements, so you keep your TRL mental model while gaining concurrency and control.

## Minimal TRL SFT example

Here's what it looks like to train **multiple configurations concurrently** on a single GPU:

```python
from rapidfireai import Experiment
from rapidfireai.automl import List, RFGridSearch, RFModelConfig, RFLoraConfig, RFSFTConfig
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup: load your dataset and define formatting
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
train_dataset = dataset["train"].select(range(128)).shuffle(seed=42)

def formatting_function(row):
    return {
        "prompt": [
            {"role": "system", "content": "You are a helpful customer support assistant."},
            {"role": "user", "content": row["instruction"]},
        ],
        "completion": [{"role": "assistant", "content": row["response"]}]
    }

# Define multiple configs to compare
config_set = List([
    RFModelConfig(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        peft_config=RFLoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"]),
        training_args=RFSFTConfig(learning_rate=1e-3, max_steps=128, fp16=True),
        formatting_func=formatting_function,
    ),
    RFModelConfig(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        peft_config=RFLoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"]),
        training_args=RFSFTConfig(learning_rate=1e-4, max_steps=128, fp16=True),
        formatting_func=formatting_function,
    )
])

# Run all configs concurrently with chunk-based scheduling
experiment = Experiment(experiment_name="sft-comparison")
config_group = RFGridSearch(configs=config_set, trainer_type="SFT")

def create_model(model_config):
    model = AutoModelForCausalLM.from_pretrained(
        model_config["model_name"], 
        device_map="auto", torch_dtype="auto", use_cache=False
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config["model_name"])
    return (model, tokenizer)

experiment.run_fit(config_group, create_model, train_dataset, num_chunks=4, seed=42)
experiment.end()
```

**What happens when you run this?**

Instead of training sequentially (Config 1 → wait → Config 2 → wait), both configs train concurrently:

| Approach | Time to Complete | GPU Utilization |
|----------|-----------------|-----------------|
| Sequential (traditional) | ~20 minutes | 50% idle time |
| RapidFire AI (concurrent) | ~2.5 minutes | 95%+ utilization |

You get comparative results **8× faster** on the same hardware. Open `http://localhost:3000` to watch live metrics and use IC Ops to stop, clone, or tweak runs in real-time based on what you're seeing.

## Benchmarks: Real-World Speedups

Here's what teams see when switching from sequential to RapidFire AI concurrent training:

| Scenario | Sequential Time | RapidFire AI Time | Speedup |
|----------|----------------|-------------------|---------|
| 4 configs, 1 GPU | 120 min | 7.5 min | **16×** |
| 8 configs, 1 GPU | 240 min | 12 min | **20×** |
| 4 configs, 2 GPUs | 60 min | 4 min | **15×** |

*Benchmarks on NVIDIA A100 40GB with TinyLlama-1.1B and Llama-3.2-1B models*

## Get Started Today

**🚀 Try it hands-on**: [Interactive Colab Notebook](http://tinyurl.com/rapidfireai-colab) — Zero setup, runs in your browser

**📚 Full Documentation**: [oss-docs.rapidfire.ai](https://oss-docs.rapidfire.ai) — Complete guides, examples, and API reference

**💻 GitHub**: [RapidFireAI/rapidfireai](https://github.com/RapidFireAI/rapidfireai) — Open source, production-ready

**📦 Install via PyPI**: [pypi.org/project/rapidfireai](https://pypi.org/project/rapidfireai) — `pip install rapidfireai`

**💬 Join the Community**: [Discord](https://discord.gg/6vSTtncKNN) — Get help, share results, request features

---

We built RapidFire AI because the status quo—training one config at a time—wastes both time and GPU cycles. Now that we're officially integrated into TRL's documentation, every TRL user can train smarter, iterate faster, and ship better models.

**Try the integration and let us know**: How much faster is your experimentation loop? What should we build next? We're just getting started, and your feedback shapes where we go from here.



