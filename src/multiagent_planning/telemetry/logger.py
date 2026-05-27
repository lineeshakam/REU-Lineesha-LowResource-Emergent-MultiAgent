import sys
import os
import time
import logging
import torch
from typing import Optional
from src.multiagent_planning.config import LOGS_DIR, WANDB_ENABLED, WANDB_PROJECT, WANDB_ENTITY

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu, himonthakur@gmail.com"
__status__ = "prototype"

logger = logging.getLogger(__name__)

class DualWrite:
    """
    Helper class to write to both stdout/stderr and a file stream.
    """
    def __init__(self, original_stream, file_stream):
        self.original_stream = original_stream
        self.file_stream = file_stream

    def write(self, data):
        self.original_stream.write(data)
        self.file_stream.write(data)
        self.file_stream.flush()

    def flush(self):
        self.original_stream.flush()
        self.file_stream.flush()


class TelemetryCapture:
    """
    Context manager to record stdin, stdout, and stderr streams,
    track peak VRAM memory consumption, and report to Weights & Biases.
    """
    def __init__(self, session_name: str, run_config: Optional[dict] = None):
        self.session_name = session_name
        self.run_config = run_config or {}
        
        self.stdout_log_path = LOGS_DIR / f"{session_name}_stdout.log"
        self.stderr_log_path = LOGS_DIR / f"{session_name}_stderr.log"
        
        self.stdout_file = None
        self.stderr_file = None
        
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        self.wandb_run = None
        self.start_time = 0.0

    def __enter__(self):
        # 1. Start stream redirection
        self.stdout_file = open(self.stdout_log_path, "w", encoding="utf-8")
        self.stderr_file = open(self.stderr_log_path, "w", encoding="utf-8")
        
        sys.stdout = DualWrite(self.original_stdout, self.stdout_file)
        sys.stderr = DualWrite(self.original_stderr, self.stderr_file)
        
        # 2. Reset CUDA Memory Tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        self.start_time = time.time()
        logger.info(f"Started telemetry session: {self.session_name}")
        logger.info(f"Stdout log file: {self.stdout_log_path}")
        logger.info(f"Stderr log file: {self.stderr_log_path}")

        # 3. Setup Wandb
        if WANDB_ENABLED:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=WANDB_PROJECT,
                    entity=WANDB_ENTITY,
                    name=self.session_name,
                    config=self.run_config
                )
                logger.info("Weights & Biases telemetry run initialized.")
            except ImportError:
                logger.warning("wandb is not installed. Skipping W&B integration.")
            except Exception as e:
                logger.error(f"Failed to initialize Wandb: {e}")

        return self

    def log_metrics(self, metrics: dict):
        """
        Logs research metrics to W&B if enabled.
        """
        if self.wandb_run:
            self.wandb_run.log(metrics)

    def get_gpu_profile(self) -> dict:
        """
        Gathers peak GPU memory statistics.
        """
        profile = {"gpu_available": torch.cuda.is_available()}
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            max_memory = torch.cuda.max_memory_allocated(0) / (1024 ** 2) # MB
            max_cached = torch.cuda.max_memory_reserved(0) / (1024 ** 2) # MB
            profile.update({
                "device_name": device_name,
                "peak_vram_allocated_mb": max_memory,
                "peak_vram_cached_mb": max_cached
            })
        return profile

    def get_profile_metrics(self) -> dict:
        """
        Returns duration and memory consumption stats for embedding inside logs.
        """
        duration = time.time() - self.start_time
        gpu_stats = self.get_gpu_profile()
        metrics = {
            "duration_sec": duration,
            "gpu_available": gpu_stats.get("gpu_available", False)
        }
        if gpu_stats.get("gpu_available"):
            metrics.update({
                "peak_vram_allocated_mb": gpu_stats["peak_vram_allocated_mb"],
                "peak_vram_cached_mb": gpu_stats["peak_vram_cached_mb"]
            })
        return metrics

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        gpu_stats = self.get_gpu_profile()
        
        logger.info(f"Telemetry session complete. Duration: {duration:.2f} seconds.")
        if gpu_stats.get("gpu_available"):
            logger.info(f"Peak VRAM usage: {gpu_stats['peak_vram_allocated_mb']:.2f} MB")
            
        # Log final profile metrics to W&B
        if self.wandb_run:
            try:
                final_metrics = {"total_duration_sec": duration}
                if gpu_stats.get("gpu_available"):
                    final_metrics.update({
                        "peak_vram_allocated_mb": gpu_stats["peak_vram_allocated_mb"],
                        "peak_vram_cached_mb": gpu_stats["peak_vram_cached_mb"]
                    })
                self.wandb_run.log(final_metrics)
                self.wandb_run.finish()
                logger.info("W&B session finalized.")
            except Exception as e:
                logger.error(f"Failed to close Wandb: {e}")

        # Restore original streams
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
        # Close file handles
        if self.stdout_file:
            self.stdout_file.close()
        if self.stderr_file:
            self.stderr_file.close()
