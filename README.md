# Parametric Copulas (GPyTorch version)

This is the GPyTorch-based package that infers copula parameters using a latent Gaussian Process model.
The package currently contains 4 copula families (Gaussian, Frank, Clayton, Gumbel) + linear combinations of copulas from same or different families.
The models are constructed with the greedy or heuristic algorithm and the best model is selected based on WAIC. 
Both greedy and heuristic algorithms perform well on synthetic data (see tests/integration).

The bivariate models can be then organised into vines (currently only C-Vine).
A number of methods for computing information measures (e.g. vine.entropy, vine.inputMI) are implemented.
