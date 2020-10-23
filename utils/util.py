def get_copula_name_string(likelihoods):
	strrot = lambda rotation: rotation if rotation is not None else ''
	copula_names=''
	for lik in likelihoods:
		copula_names += lik.name+strrot(lik.rotation)
	return copula_names

def get_vine_name(vine):
	output = ""
	for d,layer in enumerate(vine.layers):
		if d>0:
			cond = '|'
			for i in range(d):
				cond += f'{i}'	
		else:
			cond = ''
		for n,mix_model in enumerate(layer):
			name = ''
			for copula, rotation in zip(mix_model.copulas, mix_model.rotations):
				name += copula.__name__[:-6]
				if rotation!=None:
					name += rotation
			output += f"C_{d}{d+n+1}{cond}: {name}\n"
	return output