import torch
from torch import nn
from ai.ddqn import Ddqn
from entities import Match, Map, Agent
from ai.experience_replay import ReplayMemory


class Trainer:
	
	def __init__(
		self,
		model0: Ddqn,
		model1
		) -> None:
		STATE_SIZE = 1089

		self.model = model0
		self.enemy = model1
		self.memory = ReplayMemory(1000)
		self.target = Ddqn(0, STATE_SIZE, 10)
		self.loss_fn = nn.MSELoss()
		self.optimizer = None

		self.target.load_state_dict(self.model.state_dict())

				
		self.rewards_per_episode = []
		self.total_reward = 0
		self.best_reward = -999999
		self.step_count = 0
		self.previous_total_reward = 0
	
	def train(
			self, 
			epsilon_min, # the minimum value of the exploration policy
			epsilon_init, # the initial value of the exploration rate
			epsilon_decay, # the decay rate of the exploration rate, it will be exponential
			learning_rate, # the model learning rate
			discount_factor_g, # the discount factor of the rewards
			network_sync_rate, # the rate that we sync the target network with the model
			mini_batch_size = 32, # the size of the memory sample used to train the model
			continue_last_model = False):
		"""
			Receives the training parameters used and train the model
			Return:
				the total reward value, which should be maximized 
		"""
		#TODO: Add LOGS
		
		# a episode is one match
		num_actions = 10

		self.optimizer = torch.optim.Adam(
			self.model.parameters(),
			lr=learning_rate
		)
		
		epsilon = epsilon_init

		self.model.exploration_policy(True, epsilon, epsilon_decay, epsilon_min)

		for episode in range(0, 10000):

			self.m = Match(3, self.model, self.enemy, presentation=False, sleep_time=0.01, print_log=False)

			self.m.play(self.turn_callback)

			episode_reward = float(self.total_reward.item()) - self.previous_total_reward

			self.previous_total_reward = float(self.total_reward.item())

			self.rewards_per_episode.append(episode_reward)

			if episode_reward >= self.best_reward:
				self.best_reward = episode_reward
				self.save_model()

			if len(self.memory) > mini_batch_size:
				

				# sample from memory
				mini_batch = self.memory.sample(mini_batch_size)

				self.optimize(mini_batch, self.model, self.target, discount_factor_g)

				self.model.decrease_exploration_rate()

				# synching the networks
				if self.step_count > network_sync_rate:
					self.target.load_state_dict(self.model.state_dict())

		print(self.model.get_extreme_rewards())

		return self.total_reward

	def optimize(self, mini_batch, policy_dqn:Ddqn, target_dqn:Ddqn, discount_factor):
		# transpose the list of experience and separate each element
		states, actions, new_states, rewards = zip(*mini_batch)

		# stack tensors to create batch tensors
		# tensor([ [1, 2, 3] ])
		states        = torch.stack(states)
		actions       = torch.stack(actions)
		rewards       = torch.stack(rewards)
		new_states    = torch.stack(new_states)
    
		with torch.no_grad():

			best_action_from_policy = policy_dqn(new_states).argmax(dim=1)

			target_q = rewards + discount_factor * \
				target_dqn(new_states).gather(dim=1, index=best_action_from_policy.unsqueeze(dim=1)).squeeze()
		
		#calculates Q values from current policy
		current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
		'''
		policy_dqn(states)    => tensor([[1, 2, 3], [4, 5, 6]])
			actions.unsqueeze(dim=1)
			.gather(1, )index actions.unsqueeze(dim=1) => 
				.squeeze => 
		'''

		# compute loss for the whole minibatch
		loss = self.loss_fn(current_q, target_q)

		# optimize the model
		self.optimizer.zero_grad()  # clear gradients
		loss.backward()            # compute gradients (backpropagation)
		self.optimizer.step()       # update network parameters i.e. weight and bias

	def turn_callback(self, team: int, ID: int, previous_pos: tuple, action, list_agents: list[Agent]):

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
			action = torch.tensor(action, dtype=torch.int64)
			
			new_state = [item for data in view.values() for item in data]
			new_state = torch.tensor(new_state, dtype=torch.float)
			
			reward = self.model.get_reward(agents=list_agents)
			reward = torch.tensor(reward, dtype=torch.float)

			transition = (state, action, new_state, reward)

			self.memory.append(transition)

			self.total_reward += reward


	#TODO: IMPLEMENT
	def save_model(self):
		"""
		Save the current model in a file
		"""
		pass