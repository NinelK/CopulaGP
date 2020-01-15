# Parametric Copulas (GPyTorch version)

This is the GPyTorch-based package that inferes copula parameters using a latent Gaussian Process model.
The package currently contains 4 copula families (Gaussian, Frank, Clayton, Gumbel) + combinations of copulas from same or different families.
The models are constructed with the greedy algorithm and the best model is selected based on WAIC. 
Greedy algorithm performs well on synthetic data (see tests/integration).

## TODO

- [ ] Make a post-processing tool that loads models and checks how tuning differs for neurons vs. their dependence
- [x] Fix MI
	- [x] Use GP mean instead of the doubly-stochastic model
	- [x] Add transformation of the marginals to this package (move from data processing package)
	- [x] Make MI for marginals
- [ ] Add unittests for marginal transforms
- [ ] Add proper weight sharing to select_model to reduce memory usage
- [ ] Add simplifications of the final model
	- [ ] If model has constant 0 parameter -> substitute with Independence
	- [ ] If 2 elements have the same thetas -> add sharing
	- [ ] Add sharing for mixing parameters?
- [ ] Visualize the most frequently used models (first choice, combinations, etc.)
- [ ] (!) Implement vine copula models
- [ ] Implement Student-T copula
	- [x] Sampling with 1D icdf
	- [ ] Implement Bailey's sampling method
	- [ ] LogPDF
	- [ ] Tests
	- [ ] GPU support
	- [ ] Extend the code to infer both parameters with 2 independent GPs 
- [ ] Marginals with normalizing flows 

