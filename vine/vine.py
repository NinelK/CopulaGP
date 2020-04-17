import torch
import bvcopula

class CVine():
    '''
    This class represents copula C-Vine
    Attributes
    ----------
    N: int
        Number of variables
    inputs: int
        Input points
    layers:
        A list of layers. Each layer should contain N-1-i
        MixtureCopula models, where i is the layer's number.
        
    '''
    def __init__(self, layers, inputs, device=torch.device('cpu')):
        super(CVine, self).__init__()
        # for N variables there must be N-1 layers
        self.N = len(layers) + 1
        # get the number of inputs in the 1st model
        self.inputs = inputs
        for i,layer in enumerate(layers):
            assert len(layer) == self.N-1-i # check layer size
            for model in layer:
                assert self.inputs.shape[0]== model.theta.shape[-1]	#check the number of inputs
                assert (model.__class__ == bvcopula.distributions.MixtureCopula)
        self.layers = layers
        # ADD CHECK ON WHICH DEVICE EACH MODEL IS?
        self.device = device
        
    @staticmethod
    def _layer_transform(upper,new,copulas):
        '''
        Parameters
        ----------
        upper: list
            Higher layer, which is already transformed
        new: torch.Tensor ?
            New variable to be added on this layer
        copulas: list
            List of MixtureCopula models for this layer
        '''
        assert upper.shape[-1] == len(copulas)
        lower_layer = [new]
        for n, copula in enumerate(copulas):
            stack = torch.stack([upper[...,n],new],dim=-1)
            lower_layer.append(copula.make_dependent(stack))
        return torch.einsum('i...->...i',torch.stack(lower_layer))

    def sample(self, sample_size = torch.Size([])):
        # create uniform samples
        samples_shape = self.inputs.shape + sample_size + torch.Size([self.N])
        samples = torch.empty(size=samples_shape, device=self.device).uniform_(1e-4, 1. - 1e-4) #torch.rand(shape) torch.rand in (0,1]
        
        transformed_samples = [samples[...,-1:]]
        for copulas in self.layers[::-1]:
            upper = transformed_samples[-1]
            new_layer = self._layer_transform(upper,samples[...,self.N-upper.shape[-1]-1],copulas)
            transformed_samples.append(new_layer)
        
        return transformed_samples[::-1]

    def log_prob():
        pass