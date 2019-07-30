import unittest

import torch
import numpy as np
from numpy.testing import assert_allclose
from bvcopula.distributions import GaussianCopula, FrankCopula, ClaytonCopula, GumbelCopula, StudentTCopula

torch.manual_seed(0) 

class TestCopulaLogPDF(unittest.TestCase):

	def test_gaussian_thetas(self):
	  
		# check theta = 0, -1, 1
		gaussian_copula = GaussianCopula(torch.tensor([0,-1,1]).float())
		res = gaussian_copula.log_prob(gaussian_copula.sample().squeeze()).numpy()
		assert_allclose(res, (0, float("Inf"), float("Inf")))

	def test_gaussian_pdf(self):
		#test theta=0.5 values from mixedvines
		samples = torch.tensor([np.linspace(0, 1, 5), np.linspace(0.2, 0.8, 5)]).t().float()
		gaussian_copula = GaussianCopula(torch.tensor(np.full(5,0.5)).float())
		# Comparison values
		r_logpdf = np.array([-np.inf, 0.2165361255, 0.1438410362,
		                     0.2165361255, -np.inf])
		p_logpdf = gaussian_copula.log_prob(samples.squeeze()).numpy()
		assert_allclose(p_logpdf, r_logpdf)

	def test_frank_pdf(self):
		#test theta=5 values from mixedvines
		samples = torch.tensor([np.linspace(0, 1, 5), np.linspace(0.2, 0.8, 5)]).t().float()
		frank_copula = FrankCopula(torch.tensor(np.full(5,5)).float())
		# Comparison values
		r_logpdf = np.array([-np.inf, 0.4165775202, 0.3876837693, 0.4165775202,
		                     -np.inf])
		p_logpdf = frank_copula.log_prob(samples.squeeze()).numpy()
		assert_allclose(p_logpdf, r_logpdf,atol=1e-5)

	def test_clayton_pdf(self):
		#test theta=5 values from mixedvines
		samples = torch.tensor([np.linspace(0, 1, 5), np.linspace(0.2, 0.8, 5)]).t().float()
		clayton_copula = ClaytonCopula(torch.tensor(np.full(5,5)).float())
		# Comparison values
		r_logpdf = np.array([-np.inf, 0.7858645247, 0.9946292379,
                         0.6666753203, -np.inf])
		p_logpdf = clayton_copula.log_prob(samples.squeeze()).numpy()
		assert_allclose(p_logpdf, r_logpdf,atol=1e-5)

	def test_gumbel_pdf(self):
		#test theta=5 values from mixedvines
		samples = torch.tensor([np.linspace(0, 1, 5), np.linspace(0.2, 0.8, 5)]).t().float()
		gumbel_copula = GumbelCopula(torch.tensor(np.full(5,5)).float())
		# Comparison values
		r_logpdf = np.array([-np.inf, 0.84327586, 1.27675282, 0.7705065, -np.inf])
		p_logpdf = gumbel_copula.log_prob(samples.squeeze()).numpy()
		assert_allclose(p_logpdf, r_logpdf,atol=1e-5)

class TestCopulaSampling(unittest.TestCase):
	"""
	Checks that the Sampling is consistent with log_prob.
	"""

	@staticmethod
	def sampling_general(copula, bin_size):
		# here we compare sampled density vs copula pdf
		
		# generate some samples and make a binarized density array
		S = copula.sample(torch.Size([10000])).numpy().squeeze() # generates 100 x 100 (theta dim) samples
		S = S.reshape(-1,2)
		r_den = np.histogram2d(*S.T,bins=[bin_size,bin_size],density=True)[0]

		# fetch log_pdf for the same bins
		centre_bins = (np.mgrid[0:bin_size,0:bin_size]/bin_size + 1/2/bin_size).T
		samples = centre_bins.reshape(-1,2)
		samples = torch.tensor(samples).float()
		p_pdf = np.exp(copula.log_prob(samples).numpy())
		p_den = p_pdf.reshape(bin_size,bin_size)

		assert_allclose(r_den[1:-1,1:-1],p_den[1:-1,1:-1],rtol=0.05, atol=0.1)		# test each element
		assert_allclose(np.mean(r_den[1:-1,1:-1]-p_den[1:-1,1:-1]),0.,atol=0.005)	# test diff between means

		#test symmetries
		assert_allclose(r_den[1:-1,1:-1],r_den[1:-1,1:-1].T, atol=0.1)		# test each element
		assert_allclose(p_den[1:-1,1:-1],p_den[1:-1,1:-1].T, atol=0.1)		# test each element
		assert_allclose(np.mean(r_den[1:-1,1:-1]-r_den[1:-1,1:-1].T),0.,atol=0.005)	# test diff between means		
		assert_allclose(np.mean(p_den[1:-1,1:-1]-p_den[1:-1,1:-1].T),0.,atol=0.005)	# test diff between means		

	def test_gaussian_sampling(self):
		bin_size = 20
		gaussian_copula = GaussianCopula(torch.tensor(np.full(bin_size**2,0.5)).float())#torch.ones(100)*0.7)
		self.sampling_general(gaussian_copula, bin_size)

	def test_frank_sampling(self):
		bin_size = 20
		frank_copula = FrankCopula(torch.tensor(np.full(bin_size**2,5.0)).float())#torch.ones(100)*0.7)
		self.sampling_general(frank_copula, bin_size)

	def test_clayton_sampling(self):
		bin_size = 20
		clayton_copula = ClaytonCopula(torch.tensor(np.full(bin_size**2,2.0)).float())#torch.ones(100)*0.7)
		self.sampling_general(clayton_copula, bin_size)

	def test_gumbel_sampling(self):
		bin_size = 20
		gumbel_copula = GumbelCopula(torch.tensor(np.full(bin_size**2,2.0)).float())#torch.ones(100)*0.7)
		self.sampling_general(gumbel_copula, bin_size)

	def test_studentT_sampling(self):
		bin_size = 20
		student_copula = StudentTCopula(torch.tensor(np.full(bin_size**2,0.5)).float())#torch.ones(100)*0.7)
		self.sampling_general(student_copula, bin_size)

@unittest.skipUnless(torch.cuda.device_count()>0, "requires GPU")
class TestCopulaLogPDF_CUDA(unittest.TestCase):

	def test_gaussian_thetas_cuda(self):
	  
		# check theta = 0, -1, 1
		gaussian_copula = GaussianCopula(torch.tensor([0,-1,1]).cuda().float())
		res = gaussian_copula.log_prob(gaussian_copula.sample().squeeze()).cpu().numpy()
		assert_allclose(res, (0, float("Inf"), float("Inf")))

	def test_gaussian_pdf_cuda(self):
		#test theta=0.5 values from mixedvines
		samples = torch.tensor([np.linspace(0, 1, 5), np.linspace(0.2, 0.8, 5)]).t().float().cuda()
		# Gaussian copula family
		gaussian_copula = GaussianCopula(torch.tensor(np.full(5,0.5)).float().cuda())
		# Comparison values
		r_logpdf = np.array([-np.inf, 0.2165361255, 0.1438410362,
		                     0.2165361255, -np.inf]).astype("float32")
		p_logpdf = gaussian_copula.log_prob(samples.squeeze()).cpu().numpy()
		assert_allclose(p_logpdf, r_logpdf, atol=1e-6)

	def test_frank_pdf_cuda(self):
		#test theta=0.5 values from mixedvines
		samples = torch.tensor([np.linspace(0, 1, 5), np.linspace(0.2, 0.8, 5)]).t().float().cuda()
		# Gaussian copula family
		frank_copula = FrankCopula(torch.tensor(np.full(5,5)).float().cuda())
		# Comparison values
		r_logpdf = np.array([-np.inf, 0.4165775202, 0.3876837693, 0.4165775202,
		                     -np.inf]).astype("float32")
		p_logpdf = frank_copula.log_prob(samples.squeeze()).cpu().numpy()
		assert_allclose(p_logpdf, r_logpdf, atol=1e-5)

	def test_clayton_pdf_cuda(self):
		#test theta=0.5 values from mixedvines
		samples = torch.tensor([np.linspace(0, 1, 5), np.linspace(0.2, 0.8, 5)]).t().float().cuda()
		# Gaussian copula family
		clayton_copula = ClaytonCopula(torch.tensor(np.full(5,5)).float().cuda())
		# Comparison values
		r_logpdf = np.array([-np.inf, 0.7858645247, 0.9946292379,
                         0.6666753203, -np.inf]).astype("float32")
		p_logpdf = clayton_copula.log_prob(samples.squeeze()).cpu().numpy()
		assert_allclose(p_logpdf, r_logpdf, atol=1e-5)

	def test_gumbel_pdf_cuda(self):
		#test theta=0.5 values from mixedvines
		samples = torch.tensor([np.linspace(0, 1, 5), np.linspace(0.2, 0.8, 5)]).t().float().cuda()
		# Gaussian copula family
		gumbel_copula = GumbelCopula(torch.tensor(np.full(5,5)).float().cuda())
		# Comparison values
		r_logpdf = np.array([-np.inf, 0.84327586, 1.27675282, 0.7705065, -np.inf]).astype("float32")
		p_logpdf = gumbel_copula.log_prob(samples.squeeze()).cpu().numpy()
		assert_allclose(p_logpdf, r_logpdf, atol=1e-5)


@unittest.skipUnless(torch.cuda.device_count()>0, "requires GPU")
class TestCopulaSampling_CUDA(unittest.TestCase):
	"""
	Checks that the Sampling is consistent with log_prob.
	"""

	@staticmethod
	def sampling_general_GPU(copula, bin_size):
		# here we compare sampled density vs copula pdf
		bin_size = 20
	
		# generate some samples and make a binarized density array
		S = copula.sample(torch.Size([10000])).cpu().numpy().squeeze() # generates 100 x 100 (theta dim) samples
		S = S.reshape(-1,2)
		r_den = np.histogram2d(*S.T,bins=[bin_size,bin_size],density=True)[0]

		# fetch log_pdf for the same bins
		centre_bins = (np.mgrid[0:bin_size,0:bin_size]/bin_size + 1/2/bin_size).T
		samples = centre_bins.reshape(-1,2)
		samples = torch.tensor(samples).float().cuda()
		p_pdf = np.exp(copula.log_prob(samples).cpu().numpy())
		p_den = p_pdf.reshape(bin_size,bin_size)

		assert_allclose(r_den[1:-1,1:-1],p_den[1:-1,1:-1],atol=0.05)				# test each element
		assert_allclose(np.mean(r_den[1:-1,1:-1]-p_den[1:-1,1:-1]),0.,atol=1e-3)	# test diff between means

	def test_gaussian_sampling_cuda(self):
		bin_size = 20
		gaussian_copula = GaussianCopula(torch.tensor(np.full(bin_size**2,0.5)).float().cuda())#torch.ones(100)*0.7)
		self.sampling_general_GPU(gaussian_copula, bin_size)

	def test_frank_sampling_cuda(self):
		bin_size = 20
		frank_copula = FrankCopula(torch.tensor(np.full(bin_size**2,0.5)).float().cuda())#torch.ones(100)*0.7)
		self.sampling_general_GPU(frank_copula, bin_size)

	def test_clayton_sampling_cuda(self):
		bin_size = 20
		clayton_copula = ClaytonCopula(torch.tensor(np.full(bin_size**2,0.5)).float().cuda())#torch.ones(100)*0.7)
		self.sampling_general_GPU(clayton_copula, bin_size)

	def test_gumbel_sampling_cuda(self):
		bin_size = 20
		gumbel_copula = ClaytonCopula(torch.tensor(np.full(bin_size**2,0.5)).float().cuda())#torch.ones(100)*0.7)
		self.sampling_general_GPU(gumbel_copula, bin_size)