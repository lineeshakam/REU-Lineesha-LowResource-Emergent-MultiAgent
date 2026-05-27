import logging
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu, himonthakur@gmail.com"
__status__ = "prototype"

logger = logging.getLogger(__name__)

class SimplePolicyNetwork(nn.Module):
    """
    Scaffolding for policy/action probability estimation in research projects.
    Can be used to map planning states to specific coordination actions.
    """
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class RLPolicyTrainer:
    """
    Reinforcement Learning manager that optimizes planning policies using Policy Gradients (REINFORCE).
    """
    def __init__(self, policy_net: nn.Module, lr: float = 1e-3, gamma: float = 0.99):
        self.policy_net = policy_net
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma

    def train_step(self, trajectory: List[Tuple[torch.Tensor, int, float]]):
        """
        Executes policy gradient optimization step.
        trajectory: List of (state_tensor, action_index, reward)
        """
        states, actions, rewards = zip(*trajectory)

        # Compute discounted returns
        discounted_returns = []
        cumulative_reward = 0
        for r in reversed(rewards):
            cumulative_reward = r + self.gamma * cumulative_reward
            discounted_returns.insert(0, cumulative_reward)

        # Convert to tensors
        states_tensor = torch.stack(states)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        returns_tensor = torch.tensor(discounted_returns, dtype=torch.float32)

        # Normalize returns for stability
        if len(returns_tensor) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)

        # Forward pass
        action_probs = self.policy_net(states_tensor)
        
        # Gather probabilities of selected actions
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions_tensor)

        # Compute loss: -log_prob * return
        loss = -(log_probs * returns_tensor).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        logger.info(f"RL policy update complete. Loss: {loss.item():.4f}, Mean Return: {returns_tensor.mean().item():.4f}")
        return loss.item()
