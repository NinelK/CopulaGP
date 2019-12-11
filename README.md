# Parametric Copulas (GPyTorch version)

This is the GPyTorch-based package that inferes copula parameters using a latent Gaussian Process model.
The package currently contains 4 copula families (Gaussian, Frank, Clayton, Gumbel) + combinations of copulas from same or different families.
The models are constructed with the greedy algorithm and the best model is selected based on WAIC. 
Greedy algorithm performs well on synthetic data (see tests/integration).

## TODO

- [x] Add Independence to model selection
	- [x] Test Independence
	- [ ] Add proper weight sharing to reduce memory usage
- [ ] Fix MI
	- [ ] Use GP mean instead of the doubly-stochastic model
	- [x] Add transformation of the marginals to this package (move from data processing package)
	- [x] Make MI/FI for marginals
	- [ ] Add unittests for marginal transforms
	- [ ] Make a post-processing tool that loads models and checks how tuning differs for neurons vs. their dependence
- [ ] Add simplifications of the final model
	- [ ] If model has constant 0 parameter -> substitute with Independence
	- [ ] If 2 elements have the same thetas -> add sharing
	- [ ] Add sharing for mixing parameters?
- [x] Check computation_time(max_waic), as it looks dependent (nothing interesting)
- [ ] Visualize the most frequently used models (first choice, combinations, etc.)
- [x] Wrap integration tests in pytest and add to travis
- [ ] (!) Implement vine copula models
- [ ] Implement 1-param Student-T copula
	- [x] Sampling with icdf
	- [ ] Implement Bailey's sampling method
	- [ ] LogPDF
	- [ ] Tests
	- [ ] GPU support
- [ ] (!) Implement cdf/icdf for student distribution with 2-param
	- [ ] Extend the code to infer both parameters with 2 independent GPs 
- [ ] Try infer marginals with normalizing flows 

