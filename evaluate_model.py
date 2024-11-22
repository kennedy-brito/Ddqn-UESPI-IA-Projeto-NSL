from constants import *

from entities.match import Match
from entities.map import Map

from ai.random_ai import RandomAI
from ai.ddqn import Ddqn

import torch
import os

MODEL_PATH = "model/best.pt"

state_dict = {}

if os.path.isfile(MODEL_PATH):
  state_dict = torch.load(MODEL_PATH)
else:
  raise FileNotFoundError("Best model .pt file not found!")

ddqn = Ddqn(team=MY_TEAM, state_size=STATE_SIZE, num_actions=NUM_ACTIONS)

random = RandomAI(team=ENEMY_TEAM)

print("Best model found in episode: " + str(state_dict["episode"]))
print("Best reward: " + str(state_dict["best_reward"]))


ddqn.load_state_dict(state_dict['model_state_dict'])
ddqn.eval()

wins = draws = losses = 0

Map.MAX_WIDTH = 20
Map.MAX_HEIGHT = 20

enemy_lowest_hp = 30
my_lowest_hp = 30

highest_difference = {
    "difference": 0,
    "my_hp": 30,
    "enemy_hp": 30
  }
number_of_matches = 10_000

win_match = None

for _ in range(0, number_of_matches):
  match = Match(3, ddqn, random, presentation=False)
  match.play()

  hps = match.map.total_life()
  my_hp = hps[MY_TEAM]
  enemy_hp =  hps[ENEMY_TEAM]

  if my_lowest_hp > my_hp:
    my_lowest_hp = my_hp

  if enemy_lowest_hp > enemy_hp:
    enemy_lowest_hp = enemy_hp

  if enemy_hp < 1:
    win_match = str(match)

  difference = abs(my_hp - enemy_hp)

  if difference > highest_difference["difference"]:
    highest_difference["difference"] = difference
    highest_difference["enemy_hp"] = enemy_hp
    highest_difference["my_hp"] = my_hp

  if my_hp < enemy_hp:
      losses = losses + 1
  elif my_hp == enemy_hp:
    draws = draws+1
  else:
      wins = wins + 1

win_rate = wins/(wins+draws+losses) * 100
draw_rate = draws/(wins+draws+losses) * 100
loss_rate = losses/(wins+draws+losses) * 100

print(f"Win rate: {win_rate:.2}%")
print(f"Draw rate: {draw_rate:.2}%")
print(f"Loss rate: {loss_rate:.2}%")

print("\n" * 2)

print("Highest difference of Hp: " + str(highest_difference["difference"]))
print("Enemy Hp: " + str(highest_difference["enemy_hp"]))
print("My Hp: " + str(highest_difference["my_hp"]))

print("\n" * 2)

print("Worst result of each team")
print(f"My Team lowest Hp: {my_lowest_hp}")
print(f"Enemy Team lowest Hp: {enemy_lowest_hp}")

print("\n" * 2)

print("One winner match: \n")

print(win_match)
