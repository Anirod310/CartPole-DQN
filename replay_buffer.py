import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.content = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.content.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.content, batch_size)

    def __len__(self):
        return len(self.content)


