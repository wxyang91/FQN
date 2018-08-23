import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt
seq = 2*np.sin(2*np.arange(0,3.14,0.1))
plt.figure(1)
plt.plot(seq)
f = fft(seq)
plt.figure(2)
plt.plot(f)
plt.show()