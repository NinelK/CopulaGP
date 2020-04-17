# Parametric Copulas (GPyTorch version)

This is the GPyTorch-based package that inferes copula parameters using a latent Gaussian Process model.
The package currently contains 4 copula families (Gaussian, Frank, Clayton, Gumbel) + combinations of copulas from same or different families.
The models are constructed with the greedy or heuristic algorithm and the best model is selected based on WAIC. 
Both greedy and heuristic algorithms perform well on synthetic data (see tests/integration).

## TODO

- [x] Test make_dependent
- [x] Sample from a C-vine
- [ ] C-vine entropy (estimate with a given tolerance)
- [x] Heuristics to speed up model selection
- [x] Swap WAIC sign to match the original definition.

- [ ] Add unittests for marginal transforms
- [ ] Add proper weight sharing to select_model to reduce memory usage
- [ ] Add sharing for mixing parameters in a final model?
- [ ] Implement Student-T copula
	- [x] Sampling with 1D icdf
- [ ] Marginals with normalizing flows 

