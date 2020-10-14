# Parametric Copulas (GPyTorch version)

This is the GPyTorch-based package that inferes copula parameters using a latent Gaussian Process model.
The package currently contains 4 copula families (Gaussian, Frank, Clayton, Gumbel) + combinations of copulas from same or different families.
The models are constructed with the greedy or heuristic algorithm and the best model is selected based on WAIC. 
Both greedy and heuristic algorithms perform well on synthetic data (see tests/integration).

## TODO

- [ ] Review mixture distribution
	- [ ] Check what needs value expansion now
	- [ ] Maybe gather all batch shape of thetas expansions
	- [ ] Review the range (-100,88)
- [ ] When PredictiveLL mll is working better, try it again
- [ ] Rethink model saving and loading
- [ ] Add unittests for marginal transforms
- [ ] Add proper weight sharing to select_model to reduce memory usage
		(can now be done with gpytorch 1.2 linear mixture models)
- [ ] Implement Student-T copula
	- [x] Sampling with 1D icdf
- [ ] Marginals with normalizing flows?
- [ ] Fix validate args in distributions

