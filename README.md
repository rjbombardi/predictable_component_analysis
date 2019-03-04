# predictable_component_analysis
Predictable Component Analysis as performed in the article:
Bombardi RJ, Trenary L, Pegion KV, Cash B, DelSole T, Kinter JL (2018) Seasonal predictability of summer rainfall over
South America. J. Clim., 31 (20), 8181–8195. DOI: 10.1175/JCLI-D-18-0191.1
https://journals.ametsoc.org/doi/full/10.1175/JCLI-D-18-0191.1

The "laplacians.py" function calculates Laplacian eigenfunctions, which represnt an orthogonal basis set that is ordered by
length scale. For instance, Fourier series used to decompose time series are a special case of Laplacian eigenvectors.
Eigenvectors of the Laplace operator are used to identify spatial patterns that are orthogonal with respect to an area-weighted
inner product. Analogously to time series decomposition, these spatial patterns represent the decomposition of the domain by
length scale.

The "PrCA.py" function uses the output of the laplacians.py function to calculate the Predictable Component Analysis of a set of numerical simulations. This function solves an eigenvalue problem maximizing the signal-to-noise ratio, instead of variance.


