import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import plotly.graph_objects as go

class show_graphics:
    def __init__(self, **kwargs):
        self.b_low = kwargs.get('b_low', -10)
        self.b_up = kwargs.get('b_up', 10)

    def plot_f_3D_matplolib(self, f):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        x = np.arange(self.b_low, self.b_up, 0.2)
        y = np.arange(self.b_low, self.b_up, 0.2)
        x, y = np.meshgrid(x, y)
        z = f([x,y])
        surf = ax.plot_surface(x, y, z, cmap=cm.magma_r, edgecolor='none', antialiased=False)
        plt.show()

    def plot_f_3D_plotly(self, f):
        x = np.arange(self.b_low, self.b_up, 0.2)
        y = np.arange(self.b_low, self.b_up, 0.2)
        x, y = np.meshgrid(x, y)
        z = f([x, y])
        fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
        fig.show()


def f(var):
    x1, x2 = var
    return x1**2 + x2**2 - 10*(np.cos(2*np.pi*x1) + np.cos(2*np.pi*x2)) + 20

example = show_graphics(b_low = -5, b_up = 5)
example.plot_f_3D_plotly(f)
