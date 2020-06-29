from torch import linspace, no_grad, bool, Size, Tensor, arange

def evaluate(model,device):
    # define uniform test set (optionally on GPU)
    test_x = linspace(0,1,100).to(device=device)
    with no_grad():
        output = model(test_x)
    gplink = model.likelihood.gplink_function
    # _, mixes = gplink(output.rsample(Size([1000])))
    # lowest_mixes = mixes.mean(dim=1) - mixes.std(dim=1)
    _, mixes = gplink(output.mean, normalized_thetas=False)
    return mixes

def important_copulas(model, device):
    mixes = evaluate(model,device)
    which = (mixes.mean(dim=1)>0.10).type(bool) # if at least higher than 10% on average -> significant
    return which

def reduce_model(likelihoods: list, which: Tensor) -> list:
    assert len(likelihoods)==len(which)
    idx = arange(0,len(which))[which]
    return [likelihoods[i] for i in idx]