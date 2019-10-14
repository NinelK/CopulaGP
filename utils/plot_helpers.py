from numpy import ones_like

class LatentSpacePlot():
    def __init__(self, title, xlabel, ylabel, ax):
        super(LatentSpacePlot, self).__init__()
        self.ax = ax
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

    def add_plot(self, x, y, y_max=None, y_min=None, label=None):
        self.ax.plot(x, y, label=label)
        
    def add_colorbar(self, x, colors, y_level=0):
        assert(len(x)==len(colors))
        self.ax.scatter(x,ones_like(x)*y_level,color=colors)
        
    def plot_thetas(self, x, model, no_legend=False, colors=None):
        num_thetas = model.theta.shape[0]
        assert(model.theta.dim()==2)
        for i_theta in range(num_thetas):     
            name = model.copulas[i_theta].__name__
            rotation = model.rotations[i_theta]
            self.add_plot(x, model.theta[i_theta].cpu().numpy(), label = '{} {} param'.format(name,rotation))
        if not no_legend:
            self.ax.legend()
        if colors is not None:
            # place colorbar 1% of amplitude lower than minimal value
            self.add_colorbar(x, colors, 0.99*model.theta.min().item()-0.01*model.theta.min().item())
        
    def plot_mixture(self, x, model, no_legend=False, colors=None):
        num_mixes = model.mix.shape[0]
        for i_mix in range(num_mixes):     
            name = model.copulas[i_mix].__name__
            rotation = model.rotations[i_mix]
            self.add_plot(x, model.mix[i_mix].cpu().numpy(), label = '{} {} mix'.format(name,rotation))
        if not no_legend:
            self.ax.legend()
        if colors is not None:
            # place colorbar 1% of amplitude lower than minimal value
            self.add_colorbar(x, colors, 0.99*model.mix.min().item()-0.01*model.mix.min().item())
            
    def plot(self, x, model, colors=None):
        self.plot_thetas(x, model, no_legend=True)
        self.plot_mixture(x, model, no_legend=True)
        self.ax.legend()
        if colors is not None:
            self.add_colorbar(x,colors)
            
class PlotSamples():
    def __init__(self, title, xlabel, ylabel, ax):
        super(PlotSamples, self).__init__()
        self.ax = ax
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
            
    def plot(self, model, Y, colors):
        self.ax.scatter(*Y.T, color=colors)

class PlotTraining():
    def __init__(self, fig, axes, axes_labels, y_lims=None):
        super(PlotTraining, self).__init__()
        if y_lims is None:
            y_lims = [None for x in range(len(axes))]
        assert(len(axes)==len(axes_labels))
        assert(len(axes)==len(y_lims))
        for ax, label, lim in zip(axes,axes_labels,y_lims):
            ax.set_xlabel("Epoch #")
            ax.set_ylabel(label)
            if lim is not None:
                assert(len(lim)==2)
                ax.set_ylim(lim[0],lim[1])
        self.fig = fig
        self.axes = axes
            
    def plot(self, data):
        for ax, dat in zip(self.axes,data):
            ax.plot(dat)