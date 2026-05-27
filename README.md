# Multi-Agent Planning Research Framework

A modular Python research codebase designed to evaluate, train, and execute multi-agent planning models fully **offline** using local HuggingFace weights. 

This repository supports agent reasoning patterns (like ReAct and Chain-of-Thought), tree-search-based planning (Monte Carlo Tree Search), Parameter-Efficient Fine-Tuning (PEFT/LoRA) on planning trajectories, Reinforcement Learning (RL) optimization via policy gradients, and a benchmarking suite to evaluate coordination success.

---

## Features

1. **Local Offline LLM Loading**: Load Causal LLMs from a local cache or specific folder path. Automatically configures HuggingFace offline environment variables to prevent phone-home calls.
2. **Quantization & PEFT Support**: Seamless loading of models in `8-bit` or `4-bit` (via bitsandbytes) and attachment of local LoRA adapters.
3. **Multi-Agent Coordination Simulation**: Run multi-agent planning loops where agents coordinate resources and schedule tasks.
4. **MCTS Search Planning**: Monte Carlo Tree Search trajectory discovery using offline LLMs as action selectors.
5. **PEFT Fine-Tuning Pipeline**: Standard scripts for training adapters on successful agent planning trace datasets.
6. **Reinforcement Learning Scaffolding**: Support for optimizing agent policies with Policy Gradient (REINFORCE) algorithms based on environment rewards.
7. **Benchmarking Evaluator**: Log results, step counts, total rewards, and execution durations to evaluate planning quality.

---

## Installation & Setup

### 1. Conda Environment Creation
Create and activate the environment using the provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate multiagent_planning
```

### 2. Enforcing Offline Mode
By default, the framework configures environment variables so that HuggingFace runs completely offline:
- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`
- `HF_DATASETS_OFFLINE=1`

Ensure you download model files (e.g. from HuggingFace Hub) to a local directory or place them in the default cache folder beforehand. To configure where models are stored, use the `HF_HOME` environment variable or modify the configurations in `src/multiagent_planning/config.py`.

---

## Usage Guide

The main entry point `main.py` allows you to run all parts of the framework out-of-the-box using mock models, or utilizing actual downloaded weights.

### Run All Demos (Planning, MCTS, RL, Benchmarking)
```bash
python main.py
```

### Run Specific Modules
You can run specific routines by passing the `--mode` argument:

```bash
# Cooperative agent step simulation
python main.py --mode plan

# Monte Carlo Tree Search trajectory planning
python main.py --mode mcts

# Reinforcement learning update step demonstration
python main.py --mode rl

# Performance benchmark evaluation
python main.py --mode benchmark
```

### Run with a Real Local Model
To load a real model instead of mock/testing generators, download the model weights (e.g., Llama, Mistral, or GPT-2) locally and run:

```bash
export MODEL_NAME="/absolute/path/to/downloaded/model"
python main.py --mode plan --use-real-model
```

---

## Project Structure

```
├── LICENSE                      # Apache 2.0 License
├── README.md                    # This file
├── environment.yml              # Conda environment file
├── main.py                      # Simulation runner entry point
├── docs/
│   └── development.md           # Developer guide and implementation details
└── src/
    └── multiagent_planning/
        ├── __init__.py          # Package initialization
        ├── config.py            # Global configs and offline environment settings
        ├── models/
        │   └── local_llm.py     # Local HF model wrapper (quantization & adapters)
        ├── agents/
        │   ├── base_agent.py    # Abstract agent interface
        │   └── planning_agent.py# Reasoning agent with ReAct prompts
        ├── planning/
        │   ├── environment.py   # State representation, actions & rewards
        │   └── mcts.py          # Monte Carlo Tree Search algorithm
        ├── peft/
        │   └── finetune.py      # QLoRA fine-tuning script
        ├── rl/
        │   └── ppo_trainer.py   # Policy gradient reinforcement learning
        └── benchmark/
            └── evaluator.py     # Metrics collector and evaluator
```

---

## Citation & License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

Developed for prototype research.
