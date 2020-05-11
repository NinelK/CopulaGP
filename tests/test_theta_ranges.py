import unittest

import torch
import numpy as np
from numpy.testing import assert_array_less
import bvcopula
from bvcopula import conf

torch.manual_seed(0) 

class TestExtremeThetas(unittest.TestCase):

	def test_sampling(self):
		'''
		Tests that copula models generate symmetric samples for extreme values of thetas.
		We use Rosenblatt transform for sample generation. We first generate a uniform distribution
		on a square and then transfrom only 1 coordinate using copula's conditional pdf.
		As a result, asymmetry here indicates numerical issues.
		The extreme values of thetas in configuration file were adjusted to maximise the permitted
		parameter range whilest still passing this test.
		'''

		def check_symmetry(copula_model):
			#generate samples
			S = copula_model.sample(torch.Size([1000])).numpy().squeeze() # here number of samples per 1 square bin
			S = S.reshape(-1,2)
			r_den = np.histogram2d(*S.T,bins=[bin_size,bin_size],density=True)[0]
			assert_array_less(np.max(r_den - r_den.T),[4.0])

		bin_size = 50
		check_symmetry(bvcopula.GaussianCopula(torch.full([bin_size**2],-1.).float()))
		check_symmetry(bvcopula.GaussianCopula(torch.full([bin_size**2],1.).float()))

		check_symmetry(bvcopula.FrankCopula(torch.full([bin_size**2],-conf.Frank_Theta_Sampling_Max).float()))
		check_symmetry(bvcopula.FrankCopula(torch.full([bin_size**2],conf.Frank_Theta_Sampling_Max).float()))

		check_symmetry(bvcopula.ClaytonCopula(torch.full([bin_size**2],0.).float()))
		check_symmetry(bvcopula.ClaytonCopula(torch.full([bin_size**2],conf.Clayton_Theta_Sampling_Max).float()))

		check_symmetry(bvcopula.GumbelCopula(torch.full([bin_size**2],1.).float()))
		check_symmetry(bvcopula.GumbelCopula(torch.full([bin_size**2],conf.Gumbel_Theta_Sampling_Max).float()))
