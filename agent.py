import torch
import torch.optim as optim
import random
from model import DQN
from replay_buffer import ReplayBuffer

class Agent:
    def __init__(self, state_dim, action_dim, lr, buffer_capacity):
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.memory = ReplayBuffer(buffer_capacity)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def select_action(self, state, epsilon):
        random_number = random.random()
        if random_number < epsilon:
            return random.randrange(0,2)
        else :
            state_tensor = (torch.tensor(state, dtype=torch.float32)).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    def optimize_model(self, batch_size, gamma):

        if len(self.memory) < batch_size:
            return
        
        batch = ReplayBuffer.sample(batch_size)

        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*batch)

        state_tensor = torch.tensor(batch_state, dtype=torch.float32)
        action_tensor = torch.tensor(batch_action, dtype=torch.int64).unsqueeze(1)     
        reward_tensor = torch.tensor(batch_reward, dtype=torch.float32).unsqueeze(1)      
        next_state_tensor = torch.tensor(batch_next_state, dtype=torch.float32)       
        done_tensor = torch.tensor(batch_done, dtype=torch.float32).unsqueeze(1)
