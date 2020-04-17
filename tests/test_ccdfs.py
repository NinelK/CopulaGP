# import sys
# sys.path.append('/home/nina/CopulaGP/')
import unittest

import torch
import numpy as np
from bvcopula.distributions import GaussianCopula, FrankCopula, ClaytonCopula, GumbelCopula, StudentTCopula
from bvcopula import conf

torch.manual_seed(0) 

class TestCopulaCCDFS(unittest.TestCase):
	"""
	Checks that the CCDF is consistent with PPCF.
	"""

	@staticmethod
	def check_inverse(copula):
		"""
		1. Generate samples from a copula
		2. Apply Y1 = ccdf(X,Y0) = C(Y|X)
		3. Apply Y2 = ppcf(X,Y1) = C^{-1}(Y|X)
		4. Check Y0==Y2
		"""
		S = copula.sample(torch.Size([10])) # samples x thetas x 2
		S1 = S.clone()
		S1[...,0] = copula.ccdf(S1)
		S2 = copula.ppcf(S1)

		assert torch.allclose(S2,S[...,0],atol=1e-4) # Gauss and Clayton are doing 1e-7 easily

	def test_families(self):

		self.check_inverse(GaussianCopula(torch.tensor([0,-0.5,0.5,-0.99,0.99]).float()))
		self.check_inverse(ClaytonCopula(torch.tensor([0,0.5,1.0,np.sqrt(conf.Clayton_Theta_Max),conf.Clayton_Theta_Max]).float()))
		self.check_inverse(FrankCopula(torch.tensor([0,-2.,2.,-np.sqrt(conf.Frank_Theta_Max),np.sqrt(conf.Frank_Theta_Max),-conf.Frank_Theta_Max*0.8,conf.Frank_Theta_Max*0.8]).float()))
		self.check_inverse(GumbelCopula(torch.tensor([1.,np.sqrt(conf.Gumbel_Theta_Max),conf.Gumbel_Theta_Max]).float()))
		# check rotation
		self.check_inverse(ClaytonCopula(torch.tensor([0,0.5,1.0,np.sqrt(conf.Clayton_Theta_Max),conf.Clayton_Theta_Max]).float(),rotation='90°'))

	@unittest.skipUnless(torch.cuda.device_count()>0, "requires GPU")
	def test_families_CUDA(self):

		self.check_inverse(GaussianCopula(torch.tensor([0,-0.5,0.5,-0.99,0.99]).float().cuda()))
		self.check_inverse(ClaytonCopula(torch.tensor([0,0.5,1.0,np.sqrt(conf.Clayton_Theta_Max),conf.Clayton_Theta_Max]).float().cuda()))
		self.check_inverse(FrankCopula(torch.tensor([0,-2.,2.,-np.sqrt(conf.Frank_Theta_Max),np.sqrt(conf.Frank_Theta_Max),-conf.Frank_Theta_Max*0.8,conf.Frank_Theta_Max*0.8]).float().cuda()))
		self.check_inverse(GumbelCopula(torch.tensor([1.,np.sqrt(conf.Gumbel_Theta_Max),conf.Gumbel_Theta_Max]).float().cuda()))
		# check rotation
		self.check_inverse(ClaytonCopula(torch.tensor([0,0.5,1.0,np.sqrt(conf.Clayton_Theta_Max),conf.Clayton_Theta_Max]).float().cuda(),rotation='90°'))
