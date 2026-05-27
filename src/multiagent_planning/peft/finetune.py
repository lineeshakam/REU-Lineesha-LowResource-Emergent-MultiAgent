import os
import json
import argparse
import logging
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from datasets import Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu, himonthakur@gmail.com"
__status__ = "prototype"

logger = logging.getLogger(__name__)

def export_trajectory_dataset(trajectories: List[Dict[str, Any]], export_path: str, profile_metrics: Optional[Dict[str, Any]] = None):
    """
    Trajectory Dataset Exporter: Serializes planner-worker-verifier traces to a JSONL dataset.
    Embeds profile metadata if available.
    """
    logger.info(f"Exporting {len(trajectories)} trajectories to {export_path}...")
    with open(export_path, "w", encoding="utf-8") as f:
        for entry in tqdm(trajectories, desc="Exporting trajectories to JSONL"):
            record = dict(entry)
            if profile_metrics:
                record["profile_metrics"] = profile_metrics
            f.write(json.dumps(record) + "\n")
    logger.info("Export complete.")


def train_peft_adapter(
    model_name_or_path: str,
    output_dir: str,
    training_data: list,
    epochs: int = 1,
    batch_size: int = 2,
    lr: float = 2e-4
):
    """
    Fine-tunes a local LLM on agent planning trajectories using LoRA.
    """
    logger.info("Initializing PEFT/LoRA training process...")
    
    # 1. Prepare Dataset
    dataset = Dataset.from_list(training_data)
    
    def format_prompts(example):
        text = f"Prompt: {example['prompt']}\nCompletion: {example['completion']}"
        return {"text": text}

    # Format using tqdm mapping indicator
    logger.info("Formatting dataset...")
    formatted_dataset = dataset.map(format_prompts, desc="Mapping dataset instruction prompts")

    # 2. Load Tokenizer & Model
    logger.info(f"Loading tokenizer from {model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    logger.info(f"Loading model from {model_name_or_path}...")
    device_map = "auto" if torch.cuda.is_available() else None
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map=device_map,
        torch_dtype=torch_dtype,
        local_files_only=True
    )

    # 3. Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Set Up Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        local_files_only=True
    )

    # 5. Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
    )

    # 6. Run Training
    logger.info("Starting training loop...")
    trainer.train()

    # 7. Save Adapter
    logger.info(f"Saving PEFT adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Training complete.")
