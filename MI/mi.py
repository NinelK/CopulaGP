import torch
from bvcopula.models import MultitaskGPModel

def estMI(model: MultitaskGPModel, points: torch.Tensor, f_size=5):
    '''
    Estimates mutual information between variables 
    (=negative conditioned copula entropy)

    # TODO: add confidence levels, like in entropy or stimMI in vine

    Parameters
    ----------
    points: Tensor
        Input points where MI (-entropy) is estimated.
    f_size: int
        Number of samples for GP to estimate MI mean and variance
    Returns
    -------
    MI_mean : float
        Estimate of the MI in bits for a copula, parameterised with a mean of GP.
    MIs_mean : float
        Estimate of the mean MI in bits for a copula, parameterised with a GP.
    MIs_std : float
        Estimate of the standard deviation of MI in bits for a copula, 
        parameterised with a GP.
    '''
    MIs = []
    with torch.no_grad():
        fs = model(points).rsample(torch.Size([f_size])) #[samples_f, copulas, positions]
    f_mean = model(points).mean.unsqueeze(0)
    # now add mean f to a set of f samples
    fs = torch.cat((fs,f_mean),0) #[samples_f + 1, copulas, positions]

    copula = model.likelihood(fs)
    MIs = copula.entropy()
    MI_mean = MIs[-1]
    MIs = MIs[:-1]

    return (MI_mean,MIs.mean(dim=0),MIs.std(dim=0)) 