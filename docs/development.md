# Developer Documentation: Multi-Agent Planning Research

This document outlines implementation details, design decisions, and extensibility points for researchers extending this codebase.

---

## Architecture Overview


### 1. Model Wrapper (`local_llm.py`)
`LocalLLM` handles model loading in a secure offline-first environment.
- Enforces `local_files_only=True` to prevent unexpected web connections.
- Translates simple `generate` calls into HuggingFace pipeline tokens.
- Quantization is initialized via `BitsAndBytesConfig` if `bitsandbytes` is installed.

### 2. Agents (`base_agent.py` & `planning_agent.py`)
Agents decide on next actions using unstructured prompt interfaces.
- Prompt templates are defined in `PlanningAgent.__init__`. 
- Modify the `system_prompt` attribute to implement alternative planning formats, such as plan-reflection, validation prompts, or multi-turn structured dialogues.

### 3. Environment State Space (`environment.py`)
`CoordinationEnvironment` maps agent string actions into state transitions.
- A parser maps textual allocations (e.g. `"allocate resource X to task Y"`) to the dictionary-based environment state.
- For custom planning scenarios (e.g. maze navigation, distributed scheduling), override the `step` method to implement alternative state logic.

### 4. MCTS Planner (`mcts.py`)
Runs look-ahead simulation planning.
- **Selection**: Traverses current nodes using Upper Confidence bound for Trees (UCT).
- **Expansion**: Calls agents to propose decisions and queries the simulation transition environment to generate successor states.
- **Simulation**: Approximates node quality using a rollout heuristic or custom value network.
- **Backpropagation**: Distributes step rewards back to ancestors.

---

## Fine-Tuning and Optimization Guides

### Parameter-Efficient Fine-Tuning (PEFT)
To fine-tune a model on successful planning runs:
1. Export successful trajectories from `PlanningEvaluator` into a JSON Lines file containing `prompt` and `completion` fields.
2. Run `peft/finetune.py` to adapt the base LLM weights using LoRA:
```bash
python -m src.multiagent_planning.peft.finetune \
  --model_path /path/to/local/base_model \
  --output_path /path/to/peft_adapter
```
3. Load the resulting adapter with the `adapter_path` argument in `LocalLLM`.

### Reinforcement Learning (RL) Policy Optimization
Use `rl/ppo_trainer.py` to optimize agent planning outputs.
- To use RL training in your environment, formulate states as numeric tensors.
- Collect trajectory samples `(state_tensor, action_index, reward)` over episodes.
- Execute `RLPolicyTrainer.train_step` to calculate policy losses and update policy networks.
