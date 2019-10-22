def Plot_Copula_Density(axes, X, Y, interval_ends, shade=False, color='C0', mrg=.2):
    '''
        Plot Latent Space functions for synthetic data.
    '''
    import seaborn as sns
    assert len(interval_ends)-1 == len(axes)
    titles = ["{}-{}".format(s,e) for s,e in zip(interval_ends[:-1],interval_ends[1:])]
    for s,e,ax,name in zip(interval_ends[:-1],interval_ends[1:],axes.flatten(),titles):
        sns.kdeplot(*Y[(X[:]>=s) & (X[:]<e)].T, ax=ax, 
                    shade=shade,  shade_lowest=False, 
                    alpha=0.7, color=color, n_levels=6)
        ax.set(title=name, xlim=(-mrg,1+mrg), ylim=(-mrg,1+mrg))

def Plot_MixModel_Param(ax,model,test_x,x,rho=None,title=''):
    '''
        Plot Thetas and Mixing parameters of a trained GP-Copula model.
        May show strange confidence intervals due to gp_link transformaiton.
        MCMC version, in contrast, first applies gp_link and THEN computes confidence intervals.
        Parameters
        ----------
        ax : list of matplotlib.axes._subplots.AxesSubplot
            Two axes for plotting thetas and mixing parameters
        model : Mixed_GPInferenceModel
            Copula-GP model
        test_x : Tensor
            Model input values. Normally scaled to range (0,1) and could be on GPU.
        x : numpy.array
            Corresponding labels for the input values. 
        rho : numpy.array
            (x,rho) for Pearson's rho plot. Useful to compare with theta parameters.
    '''
    from torch import no_grad
    assert len(ax)==2
    
    def plot_gps(mean,low,upp, axis, skip_ind=False):
        for m,l,u,c,r in zip(mean,low,upp,copulas,rotations):
            F_mean = m.detach().cpu().numpy()
            if r==None:
                label = '{}'.format(c)
            else:
                label = '{} {}'.format(c,r)
            if skip_ind & (c == 'Independence'):
                axis.plot([],[]) #just skip 1 color to be even with mixing parameters
            else:
                line, = axis.plot(x, F_mean, label = label, alpha=0.5)
                axis.fill_between(x, l.detach().cpu().numpy(),
                            u.detach().cpu().numpy(), color=line.get_color(), alpha=0.3)

    model.eval()
    with no_grad():
        output = model(test_x)
    gplink = model.likelihood.gplink_function
                
    thetas, mixes = gplink(output.mean)
    lower, upper = output.confidence_region() #thetas & mix together
    thetas_low, mixes_low = gplink(lower)
    thetas_upp, mixes_upp = gplink(upper)
    
    likelihoods = model.likelihood.likelihoods
    copulas = [lik.name for lik in likelihoods]
    rotations = [lik.rotation for lik in likelihoods]

    plot_gps(thetas,thetas_low,thetas_upp,ax[0],skip_ind=True)
    plot_gps(mixes,mixes_low,mixes_upp,ax[1])
    
    if rho is not None:
        assert len(rho) == 2
        assert rho[0].shape == rho[1].shape
        ax[0].plot(rho[0],rho[1], '--', color='grey', label='Pearson\'s rho')

    ax[0].set_ylabel(r'$\theta$')
    ax[0].set_title('Copula parameters'+title)
    ax[1].set_ylabel('[c]')
    ax[1].set_title('Mixing param'+title)
    
    for axis in ax:
        axis.set_xlabel('Position, [cm]')
        axis.set_xlim(x.min(),x.max())
        axis.legend()

def Plot_MixModel_Param_MCMC(ax,model,test_x,x,rho=None,title='',particles=1000):
    '''
        Plot Thetas and Mixing parameters of a trained GP-Copula model.
        MCMC_version
    '''
    from torch import no_grad, Size
    from numpy import mean, std
    assert len(ax)==2
    
    def plot_gps(sampled, axis, skip_ind=False):
        for i, (t,c,r) in enumerate(zip(sampled,copulas,rotations)):
            if r==None:
                label = '{}'.format(c)
            else:
                label = '{} {}'.format(c,r)
            if skip_ind & (c == 'Independence'):
                axis.plot([],[]) #just skip 1 color to be even with mixing parameters
            else:
                sm = mean(sampled[i].cpu().numpy(),axis=0)
                line, = axis.plot(x, sm, label = label)
                stdev = std(sampled[i].cpu().numpy(),axis=0)
                axis.fill_between(x, sm-2*stdev, sm+2*stdev, color=line.get_color(), alpha=0.3)
            
    model.eval()
    with no_grad():
        output = model(test_x)
    gplink = model.likelihood.gplink_function
    thetas, mixes = gplink(output.rsample(Size([particles])))
    #MC sampling is better here, since gplink function at least for mixing 

    likelihoods = model.likelihood.likelihoods
    copulas = [lik.name for lik in likelihoods]
    rotations = [lik.rotation for lik in likelihoods]

    plot_gps(thetas,ax[0],skip_ind=True)
    plot_gps(mixes,ax[1])
    
    if rho is not None:
        assert len(rho) == 2
        assert rho[0].shape == rho[1].shape
        ax[0].plot(rho[0],rho[1], '--', color='grey', label='Pearson\'s rho')

    ax[0].set_ylabel(r'$\theta$')
    ax[0].set_title('Copula parameters'+title)
    ax[1].set_ylabel('[c]')
    ax[1].set_title('Mixing param'+title)
    
    for axis in ax:
        axis.set_xlabel('Position, [cm]')
        axis.set_xlim(x.min(),x.max())
        axis.legend()

class LatentSpacePlot():
    '''
        Plot Latent Space functions for synthetic data.
    '''
    def __init__(self, title, xlabel, ylabel, ax):
        super(LatentSpacePlot, self).__init__()
        self.ax = ax
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

    def add_plot(self, x, y, y_max=None, y_min=None, label=None):
        self.ax.plot(x, y, label=label)
        
    def add_colorbar(self, x, colors, y_level=0):
        from numpy import ones_like
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