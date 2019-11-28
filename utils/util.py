def strrot(rotation):
	if rotation is not None:
		return rotation
	else:
		return ''

def get_copula_name_string(likelihoods):
	copula_names=''
	for lik in likelihoods:
		copula_names += lik.name+strrot(lik.rotation)
	return copula_names