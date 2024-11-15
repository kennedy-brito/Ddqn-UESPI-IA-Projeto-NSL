from ai import ai
import torch, torch.nn as nn
import numpy as np
from ..entities.agent import Agent
import random as rd
class Ddqn(ai):

  def __init__(self, team: int, state_size: int, num_actions):
    super(Ddqn, self).__init__(team)

    self.exploration_policy = False
    self.epsilon = 0
    self.epsilon_decay = 0
    self.epsilon_min = 0
    self.num_action = num_actions
    self.epsilon_history = []

    self.linear = nn.Sequential(
      nn.Linear(in_feature=state_size, out_features=256),
      nn.ReLU(inplace=True),

      nn.Linear(in_feature=256, out_features=128),
      nn.ReLU(inplace=True),

      nn.Linear(in_feature=128, out_feature=num_actions)
    )

  def get_action(self, view: dict):

    """
    It does more than get the action, it memorizes the state it was 
    before the action to use that in the trainer class 
    """
    self.state = [item for data in view.values() for item in data]
    
    state = torch.tensor(self.state, dtype=torch.float32).unsqueeze(dim=0)


    if(self.exploration_policy):
      if rd.random() < self.epsilon:
        action = rd.randint(0, self.num_action - 1)
        action = torch.tensor(action, dtype=torch.int64)
      else:
        with torch.no_grad():
          q_values = self.linear(state)
        action = q_values.squeeze().argmax()

    return action
  
  def turn_reward(self, team: int, action: int, list_agents: list) -> None:
    return self.get_reward(agents=list_agents)
    
  def get_reward(self, agents: list[Agent], *args):
    """
    The reward function will be based in the enemy team total health and experience
    """

    enemy_total_health = 0
    my_total_health = 0
    enemy_total_experience = 0
    my_total_experience = 0

    for agent in agents:

      if agent.team == 0:
        my_total_health += agent.life
        my_total_experience += agent.total_exp
      else:
        enemy_total_health += agent.life
        enemy_total_experience += agent.total_exp

    reward_by_health = my_total_health - enemy_total_health
    reward_by_experience = my_total_experience - enemy_total_experience

    #TODO: normalize?
    print(f"reward by health: {reward_by_health}")
    print(f"reward by experience: {reward_by_experience}")
    print(f"total reward: {reward_by_health+reward_by_experience}")

    return reward_by_experience+reward_by_health
  
  def exploration_policy(self, active, epsilon, epsilon_decay, epsilon_min):
    
    self.exploration_policy = active

    self.epsilon        = epsilon 
    self.epsilon_min    = epsilon_min 
    self.epsilon_decay  = epsilon_decay 
    self.epsilon_history.append(self.epsilon)

  
  def decrease_exploration_rate(self):

    previous_epsilon = self.epsilon

    #TODO: Should implement an epsilon and log?
    self.epsilon = max(previous_epsilon * self.epsilon_decay, self.epsilon_min)

    self.epsilon_history.append(self.epsilon)
