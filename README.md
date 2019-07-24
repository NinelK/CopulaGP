# Parametric Copulas (GPyTorch version)

This is the GPyTorch version of the code, that inferes copula parameters using GP-SVI.

## TODO

- [x] Make Gaussian copula work
- [x] Add Frank Copula
	- [ ] GPU support
- [x] Add Clayton Copula
	- [ ] Tests
	- [ ] GPU support
	- [ ] Add Clayton rotation
- [ ] Add Gumbel Copula
	- [ ] Tests
	- [ ] GPU support
- [ ] Make model selection
- [ ] Implement 1-param Student-T copula
	- [x] Sampling
	- [ ] LogPDF
	- [ ] Tests
	- [ ] GPU support
	- [x] Maybe here transformed distribution will be faster?
		No, it won't. There is nothing to transform. There is no cdf/icdf for student distribution
- [ ] (!) Implement cdf/icdf for student distribution with 2-param
	- [ ] Extend the code to infer both parameters with 2 independent GPs
- [ ] Test on the real data
- [ ] Try infer marginals with normalizing flows 

