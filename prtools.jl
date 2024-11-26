export prtools
"""
Prtools for Julia
=================

In Prtools is defined a `prdataset` and a `prmapping`. A `prdataset` is a generalization of a matrix (annotated with object labels/targets, and feature names), and a `prmapping` is a transformation of that dataset (a classifier, a regressor, a feature mapping). 
Prtools has defined the following functions:

## Classifiers
    nmc             nearest mean classifier
    qdc             quadratic discriminant
    ldc             linear discriminant
    parzenc         Parzen classifier
    knnc            k-Nearest neighbor classifier
    mogc            Mixture of Gaussians classifier
    bayesc          bayes rule applied to densities stored in dataset
    loglc           logistic classifier
    perlc           perceptron linear classifier
## Regressors
    linearr         standard linear regression
    kNNr            k-Nearest neighbor regression
    ksmoothr        Kernel smoother
    locallinearr    Local linear regression
    kridger         Kernel ridge regression
    gpr             Gaussian Process regression
## Densities
    gaussm          Gaussian density
    parzenm         Parzen density
    knnm            k-Nearest Neighbor density
    mogm            Mixture-of-Gaussians density
## Feature reduction or transformation
    scalem          Scaling mapping
    pca             Principal component analysis
## Datasets
    gendats         simple 2D classification with 2 Gaussians
    gendatb         2D banana classification dataset
    gendatsin       1D sinusoidal regression dataset
    gendatsimple2D  2D sinusoidal regression dataset
    gendatparab     1D parabolic regression dataset
    gendatsinc      1D sinc regression dataset
## Support functions
    sqeucldistm     (squared) euclidean distances between 2 matrices
    gaussianKernel  Gaussian kernel between rows of 2 matrices
    matlabsize      size of a matrix,vector or scalar
    crossval        crossvalidation
    scatterd        scatter dataset
    plotc!          plot classifier
    plotr!          plot regression function
    plotgpr!        plot Gaussian Process regression (with std)
"""
function prtools()
    return nothing
end
