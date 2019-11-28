# Parametric Copulas (GPyTorch version)

This is the GPyTorch-based package that inferes copula parameters using a latent Gaussian Process model.
The package currently contains 4 copula families (Gaussian, Frank, Clayton, Gumbel) + combinations of copulas from same or different families.
The models are constructed with the greedy algorithm and the best model is selected based on WAIC. 
Greedy algorithm performs well on synthetic data (see tests/integration).

## TODO

- [ ] Add Independence to model selection
- [ ] Fix MI
	- [ ] Use GP mean instead of the doubly-stochastic model
	- [ ] Make MI/FI for marginals
	- [ ] Make a post-processing tool that loads models and checks how tuning differs for neurons vs. their dependence
- [ ] Check computation_time(max_waic), as it looks dependent
- [ ] Visualize the most frequently used models (first choice, combinations, etc.)
- [ ] Wrap integration tests in pytest and add to travis
- [ ] (!) Implement vine copula models
- [ ] Add transformation of the marginals to this package (move from data processing package)
- [ ] Implement 1-param Student-T copula
	- [x] Sampling with icdf
	- [ ] Implement Bailey's sampling method
	- [ ] LogPDF
	- [ ] Tests
	- [ ] GPU support
- [ ] (!) Implement cdf/icdf for student distribution with 2-param
	- [ ] Extend the code to infer both parameters with 2 independent GPs 
- [ ] Try infer marginals with normalizing flows 

