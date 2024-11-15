import torch
from ai.ddqn import Ddqn
from entities import Match, Map, Agent
from ai.experience_replay import ReplayMemory


class Trainer:
	
	def __init__(
		self,
		model0: Ddqn,
		model1
		) -> None:
		STATE_SIZE = 121*8

		self.model = model0
		self.enemy = model1
		self.memory = ReplayMemory(1000)
		self.target = Ddqn(STATE_SIZE, 10)

		self.target.load_state_dict(self.model.state_dict())

				
		self.rewards_per_episode = []
		self.total_reward = 0
		self.best_reward = -999999
		self.step_count = 0
	
	def train(
			self, 
			epsilon_min, # the minimum value of the exploration policy
			epsilon_init, # the initial value of the exploration rate
			epsilon_decay, # the decay rate of the exploration rate, it will be exponential
			learning_rate, # the model learning rate
			discount_factor_g, # the discount factor of the rewards
			network_sync_rate, # the rate that we sync the target network with the model
			mini_batch_size = 32, # the size of the memory sample used to train the model
			continue_last_model = False) -> float:
		"""
			Receives the training parameters used and train the model
			Return:
				the total reward value, which should be maximized 
		"""
		#TODO: Add LOGS
		#TODO: Add training logic
		
		# a episode is one match
		num_actions = 9

		self.optimizer = torch.optim.Adam(
			self.model.parameters(),
			lr=learning_rate
		)
		
		epsilon = epsilon_init

		self.model.exploration_policy(True, epsilon, epsilon_decay, epsilon_min)

		for episode in range(0, 1000):

			self.m = Match(3, self.model, self.enemy, presentation=False, sleep_time=0.01, print_log=False)

			self.m.play(self.turn_callback)

			episode_reward = self.total_reward - self.previous_total_reward

			self.rewards_per_episode.append(episode_reward)

			if episode_reward >= self.best_reward:
				self.best_reward = episode_reward
				self.save_model()

			if len(self.memory) > mini_batch_size:
				

				# sample from memory
				mini_batch = self.memory.sample(self.mini_batch_size)

				self.optimize(mini_batch, self.model, self.target, discount_factor_g)

				self.model.decrease_exploration_rate()

				# synching the networks
				if self.step_count > network_sync_rate:
					self.target.load_state_dict(self.model.state_dict())





		return self.total_reward

	#TODO: implement
	def optimize(self, mini_batch, policy_dqn, target_dqn, discount_factor):
		pass

	def turn_callback(self, team: int, ID: int, previous_pos: tuple, action: int, list_agents: list[Agent]):

		"""
		What this does is get the actual view, and use that to construct the new state
		It gets the state saved in the model, which is a previous state, and saves the previous and new total reward, they are used to calculate the episode reward
		"""
		if team == 0:
			
			self.step_count += 1

			view = None
			map = self.m.map
			for agent in list_agents:
				if agent.ID == ID: view = map.get_view(agent.pos)	
			
			state = self.model.state
			action = action
			new_state = [item for data in view.values() for item in data]
			reward = self.model.get_reward(agents=list_agents)

			transition = (state, action, new_state, reward)

			self.memory.append(transition)

			self.previous_total_reward = self.total_reward
			self.total_reward += reward


	#TODO: IMPLEMENT
	def save_model(self):
		"""
		Save the current model in a file
		"""
		pass