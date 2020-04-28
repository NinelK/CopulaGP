import torch
from bvcopula import MixtureCopula
from bvcopula import conf
from select_copula.conf import elements

def generate_thetas(likelihoods,n,device=torch.device('cpu')):
    m = len(likelihoods)
    thetas = torch.zeros(m,n,device=device)
    pi = 2*torch.asin(torch.tensor([1],device=device).float())
    for i, lik in enumerate(likelihoods):
        sin = torch.sin(torch.linspace(0,1,n,device=device)*2*pi + i*pi/m)
        if lik.name == 'Gaussian':
            thetas[i] = 0.+sin*0.7
        elif lik.name == 'Frank':
            thetas[i] = 0.+conf.Frank_Theta_Max*0.4*sin 
            # Frank gets unstable for extremely high correlations
            # And fits data worse than Clayton + Gumbel
        elif lik.name == 'Clayton':
            thetas[i] = conf.Clayton_Theta_Max/3*(torch.abs(sin)+0.2) #sin=0=independence 
            # +0.2 chosen to make it significantly different from independence
        elif lik.name == 'Gumbel':
            thetas[i] = 1.+(conf.Gumbel_Theta_Max/1.4-1.)/3*(torch.abs(sin)+0.2)
        elif lik.name == 'Independence':
            thetas[i] = 0.
        else:
            raise('Unknown copula')
            
    return thetas

def generate_mixes(likelihoods,n,device=torch.device('cpu')):
    m = len(likelihoods)
    mixes = torch.zeros(m,n,device=device)
    pi = 2*torch.asin(torch.tensor([1],device=device).float())
    for i in range(m):
        mixes[i] = (1 + torch.sin(torch.linspace(0,1,n,device=device)*2*pi + 2*pi*i/m))/m
    assert torch.all(torch.isclose(mixes.sum(dim=0),torch.ones_like(mixes[0])))
    return mixes

def basic_thetas(likelihoods,n,device=torch.device('cpu')):
    m = len(likelihoods)
    thetas = torch.zeros(m,n,device=device)
    for i, lik in enumerate(likelihoods):
        base = torch.ones(n,device=device)
        if lik.name == 'Gaussian':
            thetas[i] = 0.7*base
        elif lik.name == 'Frank':
            thetas[i] = (-2)*base
        elif lik.name == 'Clayton':
            thetas[i] = conf.Clayton_Theta_Max/4
        elif lik.name == 'Gumbel':
            thetas[i] = 1.+(conf.Gumbel_Theta_Max/1.4-1.)/4
        else:
            raise('Unknown copula')
            
    return thetas

def create_model(mode,likelihoods,n,device=torch.device('cpu')):
    m = len(likelihoods)
    copulas = [lik.copula for lik in likelihoods]
    rotations = [lik.rotation for lik in likelihoods]
    if mode=='thetas':
        copula_model = MixtureCopula(generate_thetas(likelihoods,n,device=device),
                    torch.ones(m,n,device=device)*(1/m),
                    copulas,
                    rotations=rotations)
    elif mode=='mixes':
        copula_model = MixtureCopula(basic_thetas(likelihoods,n,device=device),
                    generate_mixes(likelihoods,n,device=device),
                    copulas,
                    rotations=rotations)
    else:
        raise('Unknown model generation mode')
    return copula_model

# generate a random vine

def _get_random_mixture(max_el=5):
    m = 1+torch.randint(max_el,(1,))
    return [elements[i] for i in torch.randperm(len(elements))[:m]]

def get_random_vine(N, x, max_el=5, device=torch.device('cpu')):
    from vine import CVine
    layers = []
    for i in range(N-1):
        layer = []
        for j in range(N-i-1):
            layer.append(create_model('thetas',_get_random_mixture(max_el=max_el),x.numel(),device=device))
        layers.append(layer)
    return CVine(layers,x,device=device)