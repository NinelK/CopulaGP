import bvcopula

# recepie for Gauss-only copula:
# 1. leave only Ind&Gauss in elements list
# 2. set max_mix=1
# 3. and voila!

# elements to select from
elements = [bvcopula.IndependenceCopula_Likelihood(),
			bvcopula.GaussianCopula_Likelihood(),
			bvcopula.FrankCopula_Likelihood(),
			bvcopula.ClaytonCopula_Likelihood(rotation='0°'),
			bvcopula.ClaytonCopula_Likelihood(rotation='90°'),
			bvcopula.ClaytonCopula_Likelihood(rotation='180°'),
			bvcopula.ClaytonCopula_Likelihood(rotation='270°'),
			bvcopula.GumbelCopula_Likelihood(rotation='0°'),
			bvcopula.GumbelCopula_Likelihood(rotation='90°'),
			bvcopula.GumbelCopula_Likelihood(rotation='180°'),
			bvcopula.GumbelCopula_Likelihood(rotation='270°')
			]

clayton_likelihoods = [bvcopula.IndependenceCopula_Likelihood(),
				bvcopula.GaussianCopula_Likelihood(),
				bvcopula.ClaytonCopula_Likelihood(rotation='0°'),
				bvcopula.ClaytonCopula_Likelihood(rotation='90°'),
				bvcopula.ClaytonCopula_Likelihood(rotation='180°'),
				bvcopula.ClaytonCopula_Likelihood(rotation='270°'),]

gumbel_likelihoods = [bvcopula.IndependenceCopula_Likelihood(),
				bvcopula.GaussianCopula_Likelihood(),
				bvcopula.GumbelCopula_Likelihood(rotation='180°'),
				bvcopula.GumbelCopula_Likelihood(rotation='270°'),
				bvcopula.GumbelCopula_Likelihood(rotation='0°'),
				bvcopula.GumbelCopula_Likelihood(rotation='90°'),]

symmetric_likelihoods = [bvcopula.GaussianCopula_Likelihood(),
						bvcopula.FrankCopula_Likelihood()]

# max number of copulas in a copula mixture
max_mix = 6

# above this waic data is independent
waic_threshold = -0.005
