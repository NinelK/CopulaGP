import unittest

import torch
import numpy as np
from numpy.testing import assert_allclose
from bvcopula.distributions import GaussianCopula, FrankCopula

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

	def test_frank_pdf(self):
		#test theta=5 values from mixedvines
		samples = torch.tensor([np.linspace(0, 1, 5), np.linspace(0.2, 0.8, 5)]).t().float()
		# Frank copula family
		frank_copula = FrankCopula(torch.tensor(np.full(5,5)).float())
		# Comparison values
		r_logpdf = np.array([-np.inf, 0.4165775202, 0.3876837693, 0.4165775202,
		                     -np.inf])
		p_logpdf = frank_copula.log_prob(samples.squeeze()).numpy()
		assert_allclose(p_logpdf, r_logpdf,atol=1e-5)


class TestCopulaSampling(unittest.TestCase):
	"""
	Checks that the Sampling is consistent with log_prob.
	"""

	def test_gaussian_sampling(self):
		# here we compare sampled density vs copula pdf

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
		p_pdf = np.exp(gaussian_copula.log_prob(samples).numpy())
		p_den = p_pdf.reshape(bin_size,bin_size)

		assert_allclose(r_den[1:-1,1:-1],p_den[1:-1,1:-1],atol=0.05)				# test each element
		assert_allclose(np.mean(r_den[1:-1,1:-1]-p_den[1:-1,1:-1]),0.,atol=0.002)	# test diff between means

	def test_frank_sampling(self):
		# here we compare sampled density vs copula pdf

		bin_size = 20
		
		# generate some samples and make a binarized density array
		frank_copula = FrankCopula(torch.tensor(np.full(bin_size**2,0.5)).float())#torch.ones(100)*0.7)
		S = frank_copula.sample(torch.Size([10000])).numpy().squeeze() # generates 100 x 100 (theta dim) samples
		S = S.reshape(-1,2)
		r_den = np.histogram2d(*S.T,bins=[bin_size,bin_size],density=True)[0]

		# fetch log_pdf for the same bins
		centre_bins = (np.mgrid[0:bin_size,0:bin_size]/bin_size + 1/2/bin_size).T
		samples = centre_bins.reshape(-1,2)
		samples = torch.tensor(samples).float()
		p_pdf = np.exp(frank_copula.log_prob(samples).numpy())
		p_den = p_pdf.reshape(bin_size,bin_size)

		assert_allclose(r_den[1:-1,1:-1],p_den[1:-1,1:-1],atol=0.05)				# test each element
		assert_allclose(np.mean(r_den[1:-1,1:-1]-p_den[1:-1,1:-1]),0.,atol=0.002)	# test diff between means

# #TODO skip these tests if no GPU is available
# class TestCopulaLogPDF_CUDA(unittest.TestCase):

# 	def test_gaussian_thetas_cuda(self):
	  
# 		# check theta = 0, -1, 1
# 		gaussian_copula = GaussianCopula(torch.tensor([0,-1,1]).cuda().float())
# 		res = gaussian_copula.log_prob(gaussian_copula.sample().squeeze()).cpu().numpy()
# 		assert_allclose(res, (0, float("Inf"), float("Inf")))

# 	def test_gaussian_pdf_cuda(self):
# 		#test theta=0.5 values from mixedvines
# 		samples = torch.tensor([np.linspace(0, 1, 5), np.linspace(0.2, 0.8, 5)]).t().float().cuda()
# 		# Gaussian copula family
# 		gaussian_copula = GaussianCopula(torch.tensor(np.full(5,0.5)).float().cuda())
# 		# Comparison values
# 		r_logpdf = np.array([-np.inf, 0.2165361255, 0.1438410362,
# 		                     0.2165361255, -np.inf]).astype("float32")
# 		p_logpdf = gaussian_copula.log_prob(samples.squeeze()).cpu().numpy()
# 		assert_allclose(p_logpdf, r_logpdf, atol=1e-6)

# class TestCopulaSampling_CUDA(unittest.TestCase):
# 	"""
# 	Checks that the Sampling is consistent with log_prob.
# 	"""

# 	def test_gaussian_sampling_cuda(self):
# 		# here we compare sampled density vs copula pdf

# 		bin_size = 20
		
# 		# generate some samples and make a binarized density array
# 		gaussian_copula = GaussianCopula(torch.tensor(np.full(bin_size**2,0.5)).float().cuda())#torch.ones(100)*0.7)
# 		S = gaussian_copula.sample(torch.Size([10000])).cpu().numpy().squeeze() # generates 100 x 100 (theta dim) samples
# 		S = S.reshape(-1,2)
# 		r_den = np.histogram2d(*S.T,bins=[bin_size,bin_size],density=True)[0]

# 		# fetch log_pdf for the same bins
# 		centre_bins = (np.mgrid[0:bin_size,0:bin_size]/bin_size + 1/2/bin_size).T
# 		samples = centre_bins.reshape(-1,2)
# 		samples = torch.tensor(samples).float().cuda()
# 		p_pdf = np.exp(gaussian_copula.log_prob(samples).cpu().numpy())
# 		p_den = p_pdf.reshape(bin_size,bin_size)

# 		assert_allclose(r_den[1:-1,1:-1],p_den[1:-1,1:-1],atol=0.05)				# test each element
# 		assert_allclose(np.mean(r_den[1:-1,1:-1]-p_den[1:-1,1:-1]),0.,atol=1e-3)	# test diff between means