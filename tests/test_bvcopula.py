import unittest

import torch
import numpy as np
from numpy.testing import assert_allclose
from bvcopula.distributions import GaussianCopula

class TestCopulaLogPDF(unittest.TestCase):

	def test_gaussian_thetas(self):
	  
		# check theta = 0, -1, 1
		gaussian_copula = GaussianCopula(torch.tensor([0,-1,1]).float())
		res = gaussian_copula.log_prob(gaussian_copula.sample().squeeze()).numpy()
		assert_allclose(res, (0, float("Inf"), float("Inf")))

	def test_gaussian_pdf(self):
		#test theta=0.5 values from mixedvines
		samples = torch.tensor([np.linspace(0, 1, 5), np.linspace(0.2, 0.8, 5)]).t().float()
		# Gaussian copula family
		gaussian_copula = GaussianCopula(torch.tensor(np.full(5,0.5)).float())
		# Comparison values
		r_logpdf = np.array([-np.inf, 0.2165361255, 0.1438410362,
		                     0.2165361255, -np.inf])
		p_logpdf = gaussian_copula.log_prob(samples.squeeze()).numpy()
		assert_allclose(p_logpdf, r_logpdf)

class TestCopulaSampling(unittest.TestCase):
	"""
	Checks that the Sampling is consistent with log_prob.
	"""

	def test_gaussian_sampling(self):

		bin_size = 20
		
		# generate some samples and make a binarized density array
		gaussian_copula = GaussianCopula(torch.tensor(np.full(bin_size**2,0.5)).float())#torch.ones(100)*0.7)
		S = gaussian_copula.sample(torch.Size([10000])).numpy().squeeze() # generates 100 x 100 (theta dim) samples
		S = S.reshape(-1,2)
		r_den = np.histogram2d(*S.T,bins=[bin_size,bin_size],density=True)[0]

		# fetch log_pdf for the same bins
		centre_bins = (np.mgrid[0:bin_size,0:bin_size]/bin_size + 1/2/bin_size).T
		samples = centre_bins.reshape(-1,2)
		samples = torch.tensor(samples).float()
		p_logpdf = np.exp(gaussian_copula.log_prob(samples).numpy())
		p_den = p_logpdf.reshape(bin_size,bin_size)

		assert_allclose(r_den[1:-1,1:-1],p_den[1:-1,1:-1],atol=0.05)
		assert_allclose(np.mean(r_den[1:-1,1:-1]-p_den[1:-1,1:-1]),0.,atol=1e-3)