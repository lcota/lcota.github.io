from pylab import *
import seaborn as sb
import numpy as np

x = arange(0, 25, .1)
y = sin(x)
ion()
plot(x, y)
plot(x, cos(x))


