import bvcopula

# elements to select from
elements = [bvcopula.GaussianCopula_Likelihood(),
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

# max number of copulas in a copula mixture
max_mix = 6
