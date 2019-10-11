import torch
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from gpytorch import settings

class VariationalELBO(MarginalLogLikelihood):
    def __init__(self, likelihood, model, weights, num_data, particles=torch.Size([0]), combine_terms=True):
        """
        A special MLL designed for variational inference
        Args:
        - likelihood: (Likelihood) - the likelihood for the model
        - model: (Module) - the variational GP model
        - num_data: (int) - the total number of training data points (necessary for SGD)
        - combine_terms: (bool) - whether or not to sum the expected NLL with the KL terms (default True)
        """
        super(VariationalELBO, self).__init__(likelihood, model)
        self.combine_terms = combine_terms
        self.num_data = num_data
        self.weights = weights
        self.particles = particles # if particles == 0 -> GH, else -> MC with that number of particles
        # MC generally performes worse than GH for 1D inputs, however it might be useful for multi-dimensional/multi-task GPs

    def forward(self, variational_dist_f, target, **kwargs):
        num_batch = variational_dist_f.event_shape.numel()

        log_likelihood = self.likelihood.expected_log_prob(target, variational_dist_f, 
                                                           weights=self.weights,
                                                           particles=self.particles,
                                                           **kwargs).div(num_batch)
        kl_divergence = self.model.variational_strategy.kl_divergence()

        if kl_divergence.dim() > log_likelihood.dim():
            kl_divergence = kl_divergence.sum(-1)

        if log_likelihood.numel() == 1:
            kl_divergence = kl_divergence.sum()

        kl_divergence = kl_divergence.div(self.num_data)

        # Add any additional registered loss terms
        added_loss = torch.zeros_like(kl_divergence)
        had_added_losses = False
        for added_loss_term in self.model.added_loss_terms():
            added_loss.add_(added_loss_term.loss())
            had_added_losses = True

        if self.combine_terms:
            res = log_likelihood - kl_divergence
            for _, prior, closure, _ in self.named_priors():
                res.add_(prior.log_prob(closure()).sum().div(self.num_data))
            return res + added_loss
        else:
            log_prior = torch.zeros_like(log_likelihood)
            for _, prior, closure, _ in self.named_priors():
                log_prior.add_(prior.log_prob(closure()).sum())
            if had_added_losses:
                return log_likelihood, kl_divergence, log_prior.div(self.num_data), added_loss
            else:
                return log_likelihood, kl_divergence, log_prior.div(self.num_data)