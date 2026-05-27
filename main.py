import argparse
import sys
import logging
import torch
from src.multiagent_planning.config import (
    MODEL_NAME_OR_PATH, MODEL_BACKEND, MODEL_FAMILY, PEFT_TYPE, PARALLEL_WORKERS
)
from src.multiagent_planning.models.factory import get_model_backend
from src.multiagent_planning.agents.graph import build_planning_graph
from src.multiagent_planning.benchmark.evaluator import PlanningEvaluator
from src.multiagent_planning.peft.finetune import export_trajectory_dataset
from src.multiagent_planning.telemetry.logger import TelemetryCapture
from src.multiagent_planning.rl.ppo_trainer import SimplePolicyNetwork, RLPolicyTrainer

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu, himonthakur@gmail.com"
__status__ = "prototype"

logger = logging.getLogger("multiagent_planning")

def run_langgraph_flow(args):
    """
    Initializes and executes the Planner -> Worker -> Verifier LangGraph state machine.
    """
    logger.info("Initializing multi-agent LangGraph workflow...")
    
    # 1. Resolve model backend
    model = get_model_backend(
        backend_name=args.backend,
        model_name_or_path=args.model_name,
        peft_type=args.peft_type,
        use_mock=not args.use_real_model
    )
    
    # 2. Compile LangGraph State Graph
    graph = build_planning_graph(model)
    
    # 3. Define session configuration for telemetry logging
    config_details = {
        "goal": args.goal,
        "backend": args.backend,
        "model_family": args.model_type,
        "peft_type": args.peft_type,
        "parallel_workers": args.parallel_workers
    }
    
    # 4. Wrap execution in telemetry capture context
    session_name = f"langgraph_run_{args.backend}_{args.model_type}"
    with TelemetryCapture(session_name, run_config=config_details) as telemetry:
        initial_state = {
            "goal": args.goal,
            "plan": [],
            "tasks_completed": [],
            "verification_feedback": "",
            "approved": False,
            "iterations": 0,
            "logs": []
        }
        
        # Invoke state machine
        logger.info(f"Invoking graph with Goal: '{args.goal}'")
        final_state = graph.invoke(initial_state)
        
        print("\n================ FINAL LANGGRAPH STATE ================")
        print(f"Goal                 : {final_state.get('goal')}")
        print(f"Plan                 : {final_state.get('plan')}")
        print(f"Approved             : {final_state.get('approved')}")
        print(f"Iterations           : {final_state.get('iterations')}")
        print(f"Verifier Feedback    : {final_state.get('verification_feedback')}")
        print(f"Completed Tasks      : {final_state.get('tasks_completed')}")
        print("=======================================================\n")
        
        # 5. Export planning trajectory to dataset (Trajectory Exporter)
        trajectories = [
            {
                "prompt": f"Plan this goal: {final_state.get('goal')}",
                "completion": f"Plan: {final_state.get('plan')}. Verification: {final_state.get('verification_feedback')}"
            }
        ]
        export_trajectory_dataset(trajectories, f"logs/{session_name}_trajectories.jsonl")
        
        # Record final run details to W&B
        telemetry.log_metrics({
            "approved": final_state.get("approved", False),
            "iterations": final_state.get("iterations", 0),
            "plan_length": len(final_state.get("plan", []))
        })


def run_benchmark_suite(args):
    """
    Runs multi-session evaluator benchmarks on the LangGraph agents.
    """
    logger.info("Initializing evaluation benchmark suite...")
    model = get_model_backend(
        backend_name=args.backend,
        model_name_or_path=args.model_name,
        peft_type=args.peft_type,
        use_mock=not args.use_real_model
    )
    graph = build_planning_graph(model)
    
    # Environment placeholder (for legacy calls)
    from src.multiagent_planning.planning.environment import CoordinationEnvironment
    env = CoordinationEnvironment(tasks=[], resources=[])
    
    evaluator = PlanningEvaluator(environment=env)
    
    session_name = f"benchmark_suite_{args.backend}"
    with TelemetryCapture(session_name) as telemetry:
        results = evaluator.run_graph_benchmark(graph, goal=args.goal, sessions=args.sessions)
        
        successes = sum(1 for r in results if r["success"])
        avg_duration = sum(r["duration_sec"] for r in results) / len(results)
        
        print("\n================ BENCHMARK RUN AGGREGATES ================")
        print(f"Total sessions run             : {len(results)}")
        print(f"Success count                  : {successes}")
        print(f"Success rate                   : {successes / len(results) * 100:.2f}%")
        print(f"Average session duration       : {avg_duration:.4f} seconds")
        print("==========================================================\n")
        
        telemetry.log_metrics({
            "benchmark_sessions": len(results),
            "benchmark_success_rate": successes / len(results),
            "benchmark_avg_duration": avg_duration
        })


def run_rl_scaffolding():
    """
    Policy Gradient reinforcement learning run.
    """
    logger.info("Executing Reinforcement Learning policy update...")
    policy = SimplePolicyNetwork(state_dim=4, action_dim=3)
    trainer = RLPolicyTrainer(policy_net=policy)
    
    mock_trajectory = [
        (torch.tensor([1.0, 1.0, 0.0, 0.0]), 0, 2.0),
        (torch.tensor([0.0, 1.0, 1.0, 0.0]), 1, 2.0),
        (torch.tensor([0.0, 0.0, 2.0, 0.0]), 2, 10.0)
    ]
    
    trainer.train_step(mock_trajectory)


def main():
    parser = argparse.ArgumentParser(description="LangChain/LangGraph Multi-Agent Research Framework")
    parser.add_argument(
        "--mode",
        choices=["langgraph", "benchmark", "rl", "all"],
        default="all",
        help="Execution mode."
    )
    parser.add_argument(
        "--backend",
        choices=["transformers", "unsloth", "llamacpp"],
        default=MODEL_BACKEND,
        help="Underlying execution engine library."
    )
    parser.add_argument(
        "--model-type",
        choices=["qwen", "gemma", "gpt2"],
        default=MODEL_FAMILY,
        help="Model architecture class (for format templates)."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME_OR_PATH,
        help="Local file path or HuggingFace identifier."
    )
    parser.add_argument(
        "--peft-type",
        choices=["none", "lora", "qlora"],
        default=PEFT_TYPE,
        help="PEFT/LoRA adapter configuration type."
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=PARALLEL_WORKERS,
        help="Max workers for parallel async sub-tasks."
    )
    parser.add_argument(
        "--goal",
        type=str,
        default="Cooperative scheduling of meeting rooms and network instances.",
        help="Task goal for Planner-Worker-Verifier coordination."
    )
    parser.add_argument(
        "--sessions",
        type=int,
        default=2,
        help="Number of iterations to run during evaluation benchmarking."
    )
    parser.add_argument(
        "--use-real-model",
        action="store_true",
        help="Load real weights instead of running mock evaluations."
    )

    args = parser.parse_args()

    if args.mode in ("langgraph", "all"):
        run_langgraph_flow(args)
    if args.mode in ("benchmark", "all"):
        run_benchmark_suite(args)
    if args.mode in ("rl", "all"):
        run_rl_scaffolding()

if __name__ == "__main__":
    main()
