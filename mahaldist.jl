using LinearAlgebra

export MahalDist, fitMahal, predictMahalDist, predictMahal

# Define the most simple outlier detector: the Mahalanobis distance to
# the mean of the data.
mutable struct MahalDist
    mean
    cov_inv
    threshold
end

"""
      fitMahal(x,fracrej=0.05,λ=1e-6)

Estimate the mean and covariance matrix of the data `x`. From that compute
the Mahalanobis distances of the training data, and set the threshold
such that `fracrej` of the data is rejected.
To avoid invertible covariance matrices, a small value λ is added to
the diagonal.
"""
function fitMahal(x,fracrej=0.05,λ=1e-6)
    dim = size(x,2)
    reg = Diagonal(repeat([λ],dim))
    mn = mean(x,dims=1)
    cv = cov(x) .+ reg
    icv = inv(cv)
    dx = x.-mn
    d = sum( (dx*icv).*dx, dims=2)
    thr = dd_threshold(-d, fracrej)
    return MahalDist(mn,icv,thr)
end

"""
      predictMahalDist(model,x)

Compute the Mahalanobis distance on dataset `x`. The `model` is obtained from function `fitMahal`.
"""
function predictMahalDist(model,x)
    dx = x.-model.mean
    d = sum( (dx*model.cov_inv) .* dx, dims=2 )
    return d
end
"""
      predictMahal(model,x)

Predict if an object in dataset `x` is normal or outlier, according to the Mahalanobis distance.
The output is 1 for target, and  -1 for outlier. 
"""
function predictMahal(model,x)
    d = -predictMahalDist(model,x)
    return 2.0 .*(d.>model.threshold) .- 1.0
end


