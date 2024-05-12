import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple
import random
from AI_Player import AI_Player
from Card_Predictor_Model import card_predictor
from Card import SCORES

Transition = namedtuple("Transition", ("state", "simplified_state", "action", "next_state", "reward"))

def createTransitionTuple(state, action, next_state, reward):
    # probabilities = card_predictor(state)
    # likely_value = torch.sum(torch.tensor(SCORES) * probabilities).item()
    simplified_state = torch.cat((state[0][:6], torch.tensor([state[0][-1]])))
    return Transition(state, simplified_state, action, next_state, reward)

class ReplayMemory(object):
    """Copied verbatim from the PyTorch DQN tutorial.

    During training, observations from the replay memory are
    sampled for policy learning.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = createTransitionTuple(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)  

def optimize_model(
    optimizer: optim.Optimizer,
    policy,
    target,
    memory: ReplayMemory,
    batch_size: int,
    gamma: float,
):
    """Model optimization step, copied verbatim from the Torch DQN tutorial.
    
    Arguments:
        device {torch.device} -- Device
        optimizer {torch.optim.Optimizer} -- Optimizer
        policy {Policy} -- Policy net
        target {Policy} -- Target net
        memory {ReplayMemory} -- Replay memory
        batch_size {int} -- Number of observations to use per batch step
        gamma {float} -- Reward discount factor
    """
    if len(memory) < batch_size:
        return
    
    transitions = memory.sample(batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        dtype=torch.bool,
    )

    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy
    state_action_values = torch.stack([policy(state) for state in state_batch])
    state_action_values = state_action_values.gather(1, action_batch)
    
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size)
    states = torch.stack([target(state) for state in non_final_next_states])
    next_state_values[non_final_mask] = states.max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(
        state_action_values, expected_state_action_values.unsqueeze(1)
    )

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return loss.item()