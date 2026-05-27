import os
from pathlib import Path

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu, himonthakur@gmail.com"
__status__ = "prototype"

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Cache and model storage directories
HF_CACHE_DIR = os.getenv("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
LOCAL_MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Offline environment variable enforcement
OFFLINE_MODE = os.getenv("OFFLINE_MODE", "true").lower() in ("true", "1", "yes")

if OFFLINE_MODE:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

# Model Backends and Engines
# Backends: "transformers", "unsloth", "llamacpp"
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "transformers").lower()

# Supported Model Classes: "qwen", "gemma", "gpt2" (for testing)
MODEL_FAMILY = os.getenv("MODEL_FAMILY", "gpt2").lower()

# Path/Identifier for selected model
# Examples: "Qwen/Qwen2.5-7B-Instruct", "google/gemma-2-9b-it"
MODEL_NAME_OR_PATH = os.getenv("MODEL_NAME", "gpt2")

# PEFT Configuration Options
# Options: "none", "lora", "qlora"
PEFT_TYPE = os.getenv("PEFT_TYPE", "none").lower()
LORA_R = int(os.getenv("LORA_R", "8"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "16"))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.05"))

# Parallelism
# Pool size for parallel multi-agent executions
PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS", "4"))

# Weights & Biases Logging
WANDB_ENABLED = os.getenv("WANDB_ENABLED", "false").lower() in ("true", "1", "yes")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "multiagent-planning-research")
WANDB_ENTITY = os.getenv("WANDB_ENTITY", None)

# Planning limits
MAX_PLANNING_STEPS = 10
MCTS_SIMULATIONS = 20
MCTS_EXPLORATION_CONSTANT = 1.414
