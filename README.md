# Parametric Copulas (GPyTorch version)

This is the GPyTorch version of the code, that inferes copula parameters using GP-SVI.

## TODO

- [x] Make Gaussian copula work
- [x] Add Frank Copula
	- [x] GPU support
	- [ ] Figure out why for very high theta -> inf log marginal likelihood
- [x] Add Clayton Copula
	- [x] Tests
	- [x] GPU support
	- [ ] Add Clayton rotation
- [x] Add Gumbel Copula
	- [ ] Tests
	- [ ] GPU support
- [ ] Fix mean_prior (it overwrites somewhere around pyro SVI) 
- [ ] Add tests on copula symmetries
- [ ] Make model selection
- [ ] Implement 1-param Student-T copula
	- [x] Sampling with icdf
	- [ ] Implement Bailey's sampling method
	- [ ] LogPDF
	- [ ] Tests
	- [ ] GPU support
	- [x] Maybe here transformed distribution will be faster?
		No, it won't. There is nothing to transform. There is no cdf/icdf for student distribution in Pytorch
- [ ] (!) Implement cdf/icdf for student distribution with 2-param
	- [ ] Extend the code to infer both parameters with 2 independent GPs
- [ ] Test on the real data
- [ ] Try infer marginals with normalizing flows 
- [ ] Implement Kendall's tau in pytorch? (https://github.com/scipy/scipy/blob/v1.3.0/scipy/stats/stats.py#L3861-L4052)

