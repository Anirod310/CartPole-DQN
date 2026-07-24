import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.content = []
        self.position = 0
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):

        if len(self.content) < self.capacity:
            self.content.append(None)
            
        self.content[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.content, batch_size)

    def __len__(self):
        return len(self.content)


