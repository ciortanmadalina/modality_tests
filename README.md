# Modality tests and Kernel density estimations

When processing a large number of datasets which can potentially have different data distributions, we are confronted with the following considerations:
- Is the data distribution unimodal and if it is the case, which model best approximates it( uniform distribution, T-distribution, chi-square distribution, cauchy distribution, etc)?
- If the data distribution is multimodal, can we automatically identify the number of modes and provide more granular descriptive statistics?
- How can we estimate the probability density function of a new dataset?


This [notebook](https://github.com/ciortanmadalina/modality_tests/blob/master/kernel_density.ipynb) tackles the following subjects:

- Histograms vs probability density function approximation
- Kernel density estimations
- Choice of optimal bandwidth: Silverman/ Scott/ Grid Search Cross Validation
- Statistical tests for unimodal distributions
- DIP test for unimodality
- Identification of the number of modes of a data distribution based on the kernel density estimation