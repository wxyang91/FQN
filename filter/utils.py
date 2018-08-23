import numpy as np
import random

class buffer:
    def __init__(self, maxlen):
        self._buf = []
        self.size = 0
        self.maxlen = maxlen

    def enqueue(self, arr):
        self._buf.append(arr)
        self.size += 1
        if self.size > self.maxlen:
            self._buf.pop(0)

    def get_data(self):
        slist = []
        alist = []
        for sample in self._buf:
            s, a = sample
            slist.append(s)
            alist.append(a)
        return np.array(slist), np.array(alist)

    def reset(self):
        self.buf = []
        self.size = 0


class memory:
    def __init__(self, maxlen):
        self._maxlen = maxlen
        self.mem = []
    def receive(self, s_, s, a, r):
        self.mem.append((s_, s, a, r))
        if len(self.mem) > self._maxlen:
            self.mem.pop(0)

    def reset(self):
        self.mem = []

    def get_length(self):
        return len(self.mem)

    def sample_minibatch(self, batch_size):
        batch = min(batch_size, len(self.mem))
        return random.sample(self.mem, batch)
