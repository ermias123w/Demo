import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
from collections import deque
import pickle
import os

from ..config.config import system_config
from ..models.hybrid_models import PredictionResult

# Configure logging
logger = logging.getLogger(__name__)

class PPOAgent:
    """Proximal Policy Optimization agent for strategy learning"""
    
    def __init__(self, state_size: int, action_size: int = 3, lr: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size  # BUY, SELL, HOLD
        self.lr = lr
        
        # PPO hyperparameters
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.k_epochs = 4
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        
        # Networks
        self.policy_net = PolicyNetwork(state_size, action_size)
        self.value_net = ValueNetwork(state_size)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        
        # Memory for storing experiences
        self.memory = PPOMemory()
        
        # Performance tracking
        self.training_history = []
        
    def select_action(self, state: np.ndarray) -> Tuple[int, float]:
        """Select action using current policy"""
        state = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_probs = self.policy_net(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            
        return action.item(), action_logprob.item()
    
    def update(self):
        """Update policy and value networks using PPO"""
        if len(self.memory) < 10:  # Minimum batch size
            return
        
        # Get batch from memory
        states, actions, rewards, logprobs, is_terminals = self.memory.get_batch()
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_logprobs = torch.FloatTensor(logprobs)
        
        # Calculate discounted rewards
        discounted_rewards = self._calculate_discounted_rewards(rewards, is_terminals)
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        
        # Normalize rewards
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        
        # PPO update
        for _ in range(self.k_epochs):
            # Get current policy and value predictions
            action_probs = self.policy_net(states)
            dist = Categorical(action_probs)
            new_logprobs = dist.log_prob(actions)
            entropy = dist.entropy()
            
            state_values = self.value_net(states).squeeze()
            
            # Calculate ratio for PPO
            ratios = torch.exp(new_logprobs - old_logprobs)
            
            # Calculate advantages
            advantages = discounted_rewards - state_values.detach()
            
            # Policy loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(state_values, discounted_rewards)
            
            # Entropy loss (for exploration)
            entropy_loss = -entropy.mean()
            
            # Total loss
            total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Update networks
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()
            self.policy_optimizer.step()
            self.value_optimizer.step()
        
        # Clear memory
        self.memory.clear()
        
        # Log training info
        self.training_history.append({
            'timestamp': datetime.now(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item()
        })
        
        logger.info(f"PPO update completed. Total loss: {total_loss.item():.4f}")
    
    def _calculate_discounted_rewards(self, rewards: List[float], is_terminals: List[bool]) -> List[float]:
        """Calculate discounted rewards"""
        discounted_rewards = []
        discounted_reward = 0
        
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            discounted_rewards.insert(0, discounted_reward)
        
        return discounted_rewards
    
    def store_experience(self, state: np.ndarray, action: int, reward: float, 
                        logprob: float, is_terminal: bool = False):
        """Store experience in memory"""
        self.memory.store(state, action, reward, logprob, is_terminal)
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'training_history': self.training_history
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        self.training_history = checkpoint.get('training_history', [])

class PolicyNetwork(nn.Module):
    """Policy network for PPO"""
    
    def __init__(self, state_size: int, action_size: int):
        super(PolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.network(x)

class ValueNetwork(nn.Module):
    """Value network for PPO"""
    
    def __init__(self, state_size: int):
        super(ValueNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.network(x)

class PPOMemory:
    """Memory buffer for PPO experiences"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.logprobs = []
        self.is_terminals = []
    
    def store(self, state, action, reward, logprob, is_terminal):
        """Store experience"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.logprobs.append(logprob)
        self.is_terminals.append(is_terminal)
    
    def get_batch(self):
        """Get all stored experiences"""
        return (self.states, self.actions, self.rewards, 
                self.logprobs, self.is_terminals)
    
    def clear(self):
        """Clear memory"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.logprobs = []
        self.is_terminals = []
    
    def __len__(self):
        return len(self.states)

class QLearningAgent:
    """Q-Learning agent for discrete action spaces"""
    
    def __init__(self, state_size: int, action_size: int = 3, lr: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        
        # Q-learning hyperparameters
        self.gamma = 0.99
        self.epsilon = 0.1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Q-network
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # Update target network
        self.update_target_network()
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state)
        return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def update(self):
        """Update Q-network using experience replay"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        experiences = [self.memory[i] for i in batch]
        
        states = torch.FloatTensor([e[0] for e in experiences])
        actions = torch.LongTensor([e[1] for e in experiences])
        rewards = torch.FloatTensor([e[2] for e in experiences])
        next_states = torch.FloatTensor([e[3] for e in experiences])
        dones = torch.BoolTensor([e[4] for e in experiences])
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q-values
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        """Update target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

class QNetwork(nn.Module):
    """Q-network for Q-learning"""
    
    def __init__(self, state_size: int, action_size: int):
        super(QNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
    
    def forward(self, x):
        return self.network(x)

class ReinforcementLearningManager:
    """Main manager for reinforcement learning operations"""
    
    def __init__(self):
        self.ppo_agent = None
        self.q_agent = None
        self.experience_buffer = deque(maxlen=1000)
        self.update_counter = 0
        
        # Performance tracking
        self.performance_history = []
        self.model_weights = {
            'lstm': 0.33,
            'transformer': 0.33,
            'sentiment': 0.34
        }
        
        # Strategy parameters
        self.strategy_params = {
            'confidence_threshold': system_config.CONFIDENCE_THRESHOLD,
            'risk_multiplier': 1.0,
            'leverage_factor': 1.0
        }
    
    def initialize_agents(self, state_size: int):
        """Initialize RL agents"""
        self.ppo_agent = PPOAgent(state_size)
        self.q_agent = QLearningAgent(state_size)
        logger.info("Reinforcement learning agents initialized")
    
    def process_prediction_outcome(self, prediction: PredictionResult, 
                                 outcome: Dict) -> Dict:
        """Process prediction outcome and update models"""
        
        # Calculate reward
        reward = self._calculate_reward(prediction, outcome)
        
        # Create state representation
        state = self._create_state_representation(prediction)
        
        # Map signal to action
        action = self._signal_to_action(prediction.signal)
        
        # Store experience
        self.experience_buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'outcome': outcome,
            'prediction': prediction,
            'timestamp': datetime.now()
        })
        
        # Update counter
        self.update_counter += 1
        
        # Update models if enough experiences
        if self.update_counter >= system_config.RL_UPDATE_FREQUENCY:
            self._update_models()
            self.update_counter = 0
        
        # Update strategy parameters
        self._update_strategy_parameters(outcome)
        
        return {
            'reward': reward,
            'weights_updated': self.update_counter == 0,
            'confidence_adjusted': self._calculate_confidence_adjustment(outcome),
            'risk_updated': True,
            'new_weights': self.model_weights.copy(),
            'new_params': self.strategy_params.copy()
        }
    
    def _calculate_reward(self, prediction: PredictionResult, outcome: Dict) -> float:
        """Calculate reward based on prediction outcome"""
        
        # Base reward for correct prediction
        base_reward = 1.0 if outcome['correct'] else -0.5
        
        # Adjust for confidence
        confidence_bonus = prediction.confidence * 0.5
        
        # Adjust for profit/loss
        profit_bonus = outcome['profit_loss'] * 10  # Scale profit/loss
        
        # Risk adjustment
        risk_penalty = -abs(prediction.leverage - 1.0) * 0.1
        
        total_reward = base_reward + confidence_bonus + profit_bonus + risk_penalty
        
        return np.clip(total_reward, -2.0, 2.0)  # Clip to reasonable range
    
    def _create_state_representation(self, prediction: PredictionResult) -> np.ndarray:
        """Create state representation for RL"""
        
        # Technical features
        technical_features = prediction.technical_features
        
        # Market context features
        market_features = np.array([
            prediction.confidence,
            prediction.sentiment_score,
            prediction.leverage,
            (prediction.take_profit - prediction.entry_price) / prediction.entry_price,
            (prediction.entry_price - prediction.stop_loss) / prediction.entry_price
        ])
        
        # Strategy parameters
        strategy_features = np.array([
            self.strategy_params['confidence_threshold'],
            self.strategy_params['risk_multiplier'],
            self.strategy_params['leverage_factor']
        ])
        
        # Combine all features
        state = np.concatenate([
            technical_features,
            market_features,
            strategy_features,
            list(self.model_weights.values())
        ])
        
        return state
    
    def _signal_to_action(self, signal: str) -> int:
        """Convert signal to action index"""
        signal_map = {'SELL': 0, 'HOLD': 1, 'BUY': 2}
        return signal_map.get(signal, 1)
    
    def _action_to_signal(self, action: int) -> str:
        """Convert action index to signal"""
        action_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        return action_map.get(action, 'HOLD')
    
    def _update_models(self):
        """Update RL models with accumulated experiences"""
        
        if not self.experience_buffer:
            return
        
        # Initialize agents if needed
        if self.ppo_agent is None:
            state_size = len(self.experience_buffer[0]['state'])
            self.initialize_agents(state_size)
        
        # Process experiences for PPO
        for experience in self.experience_buffer:
            if self.ppo_agent:
                self.ppo_agent.store_experience(
                    experience['state'],
                    experience['action'],
                    experience['reward'],
                    0.0,  # logprob will be calculated
                    False  # is_terminal
                )
        
        # Update PPO agent
        if self.ppo_agent:
            self.ppo_agent.update()
        
        # Update model weights based on performance
        self._update_model_weights()
        
        logger.info("RL models updated successfully")
    
    def _update_model_weights(self):
        """Update ensemble model weights based on performance"""
        
        # Analyze recent performance by model
        recent_experiences = list(self.experience_buffer)[-50:]  # Last 50 experiences
        
        if not recent_experiences:
            return
        
        # Calculate model performance (simplified)
        model_performance = {
            'lstm': np.mean([exp['reward'] for exp in recent_experiences]),
            'transformer': np.mean([exp['reward'] for exp in recent_experiences]),
            'sentiment': np.mean([exp['reward'] for exp in recent_experiences])
        }
        
        # Update weights based on performance (with momentum)
        alpha = 0.1  # Learning rate for weight updates
        
        for model, performance in model_performance.items():
            if performance > 0:
                self.model_weights[model] += alpha * performance
            else:
                self.model_weights[model] -= alpha * abs(performance)
        
        # Normalize weights
        total_weight = sum(self.model_weights.values())
        for model in self.model_weights:
            self.model_weights[model] /= total_weight
        
        logger.info(f"Updated model weights: {self.model_weights}")
    
    def _update_strategy_parameters(self, outcome: Dict):
        """Update strategy parameters based on outcome"""
        
        # Adaptive confidence threshold
        if outcome['correct']:
            self.strategy_params['confidence_threshold'] *= 0.99  # Slightly lower threshold
        else:
            self.strategy_params['confidence_threshold'] *= 1.01  # Slightly higher threshold
        
        # Clamp threshold
        self.strategy_params['confidence_threshold'] = np.clip(
            self.strategy_params['confidence_threshold'], 0.5, 0.9
        )
        
        # Risk multiplier adjustment
        profit_loss = outcome['profit_loss']
        if profit_loss > 0:
            self.strategy_params['risk_multiplier'] *= 1.01
        else:
            self.strategy_params['risk_multiplier'] *= 0.99
        
        # Clamp risk multiplier
        self.strategy_params['risk_multiplier'] = np.clip(
            self.strategy_params['risk_multiplier'], 0.5, 2.0
        )
    
    def _calculate_confidence_adjustment(self, outcome: Dict) -> float:
        """Calculate confidence adjustment based on outcome"""
        
        if outcome['correct']:
            return 0.01  # Increase confidence
        else:
            return -0.005  # Decrease confidence
    
    def get_adjusted_prediction(self, base_prediction: PredictionResult) -> PredictionResult:
        """Get prediction adjusted by RL parameters"""
        
        from dataclasses import replace
        
        # Adjust confidence
        adjusted_confidence = base_prediction.confidence * self.strategy_params['risk_multiplier']
        adjusted_confidence = np.clip(adjusted_confidence, 0.0, 1.0)
        
        # Adjust leverage
        adjusted_leverage = base_prediction.leverage * self.strategy_params['leverage_factor']
        adjusted_leverage = np.clip(adjusted_leverage, 1.0, system_config.MAX_LEVERAGE)
        
        # Create adjusted prediction
        adjusted_prediction = replace(
            base_prediction,
            confidence=adjusted_confidence,
            leverage=adjusted_leverage
        )
        
        return adjusted_prediction
    
    def get_model_weights(self) -> Dict[str, float]:
        """Get current model weights"""
        return self.model_weights.copy()
    
    def get_strategy_parameters(self) -> Dict[str, float]:
        """Get current strategy parameters"""
        return self.strategy_params.copy()
    
    def save_rl_state(self, filepath: str):
        """Save RL state to file"""
        state = {
            'model_weights': self.model_weights,
            'strategy_params': self.strategy_params,
            'performance_history': self.performance_history,
            'update_counter': self.update_counter
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load_rl_state(self, filepath: str):
        """Load RL state from file"""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.model_weights = state.get('model_weights', self.model_weights)
            self.strategy_params = state.get('strategy_params', self.strategy_params)
            self.performance_history = state.get('performance_history', [])
            self.update_counter = state.get('update_counter', 0)
            
            logger.info("RL state loaded successfully")
        except FileNotFoundError:
            logger.warning("No saved RL state found, using defaults")
        except Exception as e:
            logger.error(f"Error loading RL state: {e}")

# Global RL manager instance
rl_manager = ReinforcementLearningManager()