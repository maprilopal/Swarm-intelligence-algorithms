import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import plotly.graph_objects as go
import plotly.express as px

class Graphics:
    def __init__(self, **kwargs):
        self.b_low = kwargs.get('b_low', -10)
        self.b_up = kwargs.get('b_up', 10)

    def plot_f_3D_matplolib(self, f):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        x = np.arange(self.b_low, self.b_up, 0.2)
        y = np.arange(self.b_low, self.b_up, 0.2)
        x, y = np.meshgrid(x, y)
        z = f([x,y])
        surf = ax.plot_surface(x, y, z, cmap=cm.winter, edgecolor='none', antialiased=False)
        plt.show()

    def plot_f_3D_plotly(self, f):
        x = np.arange(self.b_low, self.b_up, 0.2)
        y = np.arange(self.b_low, self.b_up, 0.2)
        x, y = np.meshgrid(x, y)
        z = f([x, y])
        fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='magma_r')])
        fig.show()

    def show_values(self, values):
        x = [values[i][0] for i in range(len(values))]
        y = [values[i][1] for i in range(len(values))]
        mean = np.mean(values, axis=0)
        fig = go.Figure(data=go.Scatter(x=x, y=y, mode='markers', name='values', marker_color='rgba(152,0,0,.8)'))
        fig.add_trace(go.Scatter(x=[mean[0]], y=[mean[1]], mode='markers', name='average value'))
        #fig.update_xaxes(range=[self.b_low, self.b_up])
        #fig.update_yaxes(range=[self.b_low, self.b_up])
        fig.show()

    def show_progress(self, values):
        x = [values[i][0] for i in range(len(values))]
        y = [values[i][1] for i in range(len(values))]
        mean = np.mean(values, axis=0)
        fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines+markers', name='progress', marker=dict( size=16,
                                                                                                      color=np.random.randn(500),
                                                                                                      colorscale='Viridis',
                                                                                                      showscale=True)))
        #average = [np.average(x), np.average(y)]
        fig.add_trace(go.Scatter(x=[mean[0]], y=[mean[1]], mode='markers', name='average value'))
        #fig.update_xaxes(range=[self.b_low, self.b_up])
        #fig.update_yaxes(range=[self.b_low, self.b_up])
        fig.show()






def f(var):
    x1, x2 = var
    return x1**2 + x2**2 - 10*(np.cos(2*np.pi*x1) + np.cos(2*np.pi*x2)) + 20

#example = Graphics(b_low = -5, b_up = 5)
#example.plot_f_3D_matplolib(f)
