from ai import ai
import torch, torch.nn as nn
import numpy as np
from ..entities.agent import Agent
class Ddqn(ai):

  def __init__(self, team: int, state_size: int, num_actions):
    super(Ddqn, self).__init__(team)

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


    q_values = self.linear(state)

    return q_values.squeeze().argmax().item()
  
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

    return reward_by_experience+reward_by_health