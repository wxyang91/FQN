from filter.utils import *
from numpy.fft import fft,fftn
class Agent:
    def __init__(self):
        self.buf = buffer(10)
        self.memory = memory(10000)

    def observe(self, s, a, r):
        s_, a_ = self.buf.get_data()
        self.buf.enqueue((s, a))
        cs, _ = self.buf.get_data()
        s_ = self._fft(s_)
        a_ = self._fft(a_)
        cs = self._fft(cs)
        self.memory.receive(s_, cs, a_, r)

    def _fft(self, a):
        return fftn(a, axes=1)

    def train(self):
        #TODO:
        pass
