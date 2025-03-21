from collections import namedtuple, deque
import random

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory:

    def __init__(self, capacity):
        # deque is a double ended queue
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

