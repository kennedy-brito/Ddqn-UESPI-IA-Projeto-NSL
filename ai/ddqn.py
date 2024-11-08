from ai import ai
import torch, torch.nn as nn
import numpy as np

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
     # Constrói o estado como um tensor diretamente
    state = torch.tensor(
      [float(item) for key, data in view.items() for item in [key] + data], dtype=torch.float32
      )


    q_values = self.linear(state)

    return q_values.squeeze().argmax()
  
  def turn_reward(self, team: int, action: int, list_agents: list) -> None:
    raise NotImplementedError("get turn reward not implemented")
    
  def get_reward(self, agents: dict, *args):
    raise NotImplementedError("get reward not implemented")