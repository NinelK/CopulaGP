from torch import linspace, no_grad, bool, Size, Tensor, arange

def evaluate(model):
    # define uniform test set (optionally on GPU)
    test_x = linspace(0,1,100).to(device=model.device)
    with no_grad():
        output = model.gp_model(test_x)
    # _, mixes = gplink(output.rsample(Size([1000])))
    # lowest_mixes = mixes.mean(dim=1) - mixes.std(dim=1)
    _, mixes = model.gplink(output.mean, normalized_thetas=False)
    return mixes

def important_copulas(model):
    mixes = evaluate(model)
    which = (mixes.mean(dim=1)>0.10).type(bool) # if at least higher than 10% on average -> significant
    return which

def reduce_model(bvcopulas: list, which: Tensor) -> list:
    assert len(bvcopulas)==len(which)
    idx = arange(0,len(which))[which]
    return [bvcopulas[i] for i in idx]