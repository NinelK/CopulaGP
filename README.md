# Parametric Copulas (GPyTorch version)

This is the GPyTorch version of the code, that inferes copula parameters using GP-SVI.

## TODO

- [x] Make Gaussian copula work
- [x] Add Frank Copula
- [x] Add Clayton Copula
- [x] Add Gumbel Copula
- [x] Add rotation
- [x] Add tests on copula symmetries
- [x] Make model selection
- [x] Add Mixture Model
	- [x] Mix more than 2 copulas
- [x] Test on the real data
- [ ] Implement 1-param Student-T copula
	- [x] Sampling with icdf
	- [ ] Implement Bailey's sampling method
	- [ ] LogPDF
	- [ ] Tests
	- [ ] GPU support
- [ ] (!) Implement cdf/icdf for student distribution with 2-param
	- [ ] Extend the code to infer both parameters with 2 independent GPs
- [ ] Fix mean_prior (it overwrites somewhere around pyro SVI) 
- [ ] Try infer marginals with normalizing flows 
- [ ] Implement Kendall's tau in pytorch? (https://github.com/scipy/scipy/blob/v1.3.0/scipy/stats/stats.py#L3861-L4052)

