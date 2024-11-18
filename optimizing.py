import json
from ai.ddqn import Ddqn
from ai.random_ai import RandomAI
from trainer import Trainer
from constants import *
from bayes_opt import BayesianOptimization
"""
Lógica da otimização

Criar um modelo com uma seed fixa

Treinar esse modelo com alguns hyperparametros 

Verificar a recompensa média desse treino

Testar outros hyperparametros 

Comparar as recompensas
"""
pbounds = {
  'epsilon_min': [0.01, 0.1],
  'epsilon_init' : [0.8, 1.0],
  'epsilon_decay' : [0.99, 0.9999],
  'learning_rate' : [0.0001, 0.01],
  'discount_factor_g' : [0.9,0.99],
  'network_sync_rate' : [500, 2000],
  "mini_batch_size" : [30, 600]

}
base_model: Ddqn = Ddqn(MY_TEAM, STATE_SIZE, NUM_ACTIONS)

def train(
            epsilon_min, # the minimum value of the exploration policy
			epsilon_init, # the initial value of the exploration rate
			epsilon_decay, # the decay rate of the exploration rate, it will be exponential
			learning_rate, # the model learning rate
			discount_factor_g, # the discount factor of the rewards
			network_sync_rate, # the rate that we sync the target network with the model
			mini_batch_size = 40 # the size of the memory sample used to train the model
      ):
  global base_model

  model = Ddqn(MY_TEAM, STATE_SIZE, NUM_ACTIONS)

  model.load_state_dict(base_model.state_dict())

  t = Trainer(model, RandomAI(ENEMY_TEAM))

  t.activate_log(False)

  t.episodes_trained(EXPLORATORY_TRAINING_QUANTITY)
  
  mean_reward = t.train(
    epsilon_min= epsilon_min, 
    epsilon_init=  epsilon_init, 
    epsilon_decay= epsilon_decay, 
    learning_rate= learning_rate, 
    discount_factor_g= discount_factor_g, 
    network_sync_rate= network_sync_rate,
    mini_batch_size=mini_batch_size
  )

  return mean_reward

optimizer = BayesianOptimization(
  f=train,
  pbounds=pbounds,
  random_state=1
)

optimizer.maximize(
  n_iter=50,
  init_points=10
)

print(optimizer.max)


for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))


# Salvando no arquivo de log
with open("hyperparams_normalized.json", "w") as log_file:
    json.dump(optimizer.max, log_file, indent=4)  # `indent=4` para deixar legível

print("Hyperparâmetros otimizados salvos no arquivo hyperparams_normalized.json!")