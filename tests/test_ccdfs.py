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
		# print(S[...,0])
		S1[...,0] = copula.ccdf(S1)
		# print(S1[...,0])
		S2 = copula.ppcf(S1)
		# print(S[...,0])
		# print(S2)
		# print(S2-S[...,0])
		if copula.__class__.__name__=='FrankCopula':
			assert torch.allclose(S2[...,:-1],S[...,:-1,0]) & torch.allclose(S2[...,-1],S[...,-1,0],atol=1e-3)
		elif copula.__class__.__name__=='GumbelCopula':
			assert torch.allclose(S2,S[...,0],rtol=1e-3)
		elif copula.__class__.__name__=='ClaytonCopula':
			assert torch.allclose(S2,S[...,0],atol=1e-4)
		else:
			assert torch.allclose(S2,S[...,0])

	def test_families(self):

		self.check_inverse(GaussianCopula(torch.tensor([0,-0.5,0.5,-0.99,0.99]).float()))
		self.check_inverse(ClaytonCopula(torch.tensor([0,0.5,1.0,np.sqrt(conf.Clayton_Theta_Max),conf.Clayton_Theta_Max]).float()))
		self.check_inverse(FrankCopula(torch.tensor([0,-0.5,0.5,-np.sqrt(conf.Frank_Theta_Max),np.sqrt(conf.Frank_Theta_Max),-conf.Frank_Theta_Max,conf.Frank_Theta_Max]).float()))
		self.check_inverse(GumbelCopula(torch.tensor([1.,np.sqrt(conf.Gumbel_Theta_Max),conf.Gumbel_Theta_Max]).float()))

	@unittest.skipUnless(torch.cuda.device_count()>0, "requires GPU")
	def test_families_CUDA(self):

		self.check_inverse(GaussianCopula(torch.tensor([0,-0.5,0.5,-0.99,0.99]).float().cuda()))
		self.check_inverse(ClaytonCopula(torch.tensor([0,0.5,1.0,np.sqrt(conf.Clayton_Theta_Max),conf.Clayton_Theta_Max]).float().cuda()))
		self.check_inverse(FrankCopula(torch.tensor([0,-0.5,0.5,-np.sqrt(conf.Frank_Theta_Max),np.sqrt(conf.Frank_Theta_Max),-conf.Frank_Theta_Max,conf.Frank_Theta_Max]).float().cuda()))
		self.check_inverse(GumbelCopula(torch.tensor([1.,np.sqrt(conf.Gumbel_Theta_Max),conf.Gumbel_Theta_Max]).float().cuda()))
