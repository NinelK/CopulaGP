# Parametric Copulas (GPyTorch version)

This is the GPyTorch-based package that inferes copula parameters using a latent Gaussian Process model.
The package currently contains 4 copula families (Gaussian, Frank, Clayton, Gumbel) + combinations of copulas from same or different families.
The models are constructed with the greedy algorithm and the best model is selected based on WAIC. 
Greedy algorithm performs well on synthetic data (see tests/integration).

## TODO

- [x] CCDF for a model mixture
- [x] Transform to second layer
	- [ ] Serialise
- [ ] Test make_dependent
- [ ] Sample from a C-vine
- [ ] Entropy

- [ ] Add unittests for marginal transforms
- [ ] Add proper weight sharing to select_model to reduce memory usage
- [ ] Add simplifications of the final model
	- [ ] If model has constant 0 parameter -> substitute with Independence
	- [ ] If 2 elements have the same thetas -> add sharing
	- [ ] Add sharing for mixing parameters?
- [ ] Heuristics to speed up model selection?
- [ ] Implement Student-T copula
	- [x] Sampling with 1D icdf
- [ ] Marginals with normalizing flows 

