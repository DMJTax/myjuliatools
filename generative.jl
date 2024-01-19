"""
   Generative classifiers

The classical way of defining classifiers:
1. estimate the probability density of each of the classes
2. estimate the class prior probabilities
3. use Bayes' rule to obtain the class posterior probabilities

This can be done in Julia like:
```
> a = gendatb()       # get data
  # estimate a gaussian for the first class, a Parzen for the second:
> u = generativec([gaussm() parzenm()])
> w = a*u             # train the classifier
> pred = a*w          # get the predictions
> lab = labeld(pred)  # show the predicted labels
> testc(pred)         # and the error
```

"""

using LinearAlgebra
using Statistics
using StatsBase

export sqeucldistm, matlabsize, gaussm, parzenm, knnm, qdc, parzenc, knnc, bayesc, generativem, generativec

"""
       D = sqeucldistm(A,B)
Compute the Squared Euclidean distance between the rows of matrix `A`
and the rows of matrix `B`. If `A` has size `NxD`, and `B` size `MxD`,
then output matrix is `NxM`.
"""
function sqeucldistm(A,B)
    n,dim = matlabsize(A)  # avoid problems with 1D datasets
    n2,dim2 = matlabsize(B)
    if (dim!=dim2)
        error("sqeucldistm: Matrix dimensions do not match.")
    end
    D = zeros(n,n2)
    for i=1:n
        for j=1:n2
            df = A[i,:] .- B[j,:]
            D[i,j] = df'*df
        end
    end
    return D
end
"""
    sz = matlabsize(x)
Return the size of matrix, vector or scalar `x`. When `x` is a scalar,
the size will be `1x1` (instead of `()`), and when `x` is a vector it
will be `Nx1` (instead of `(N,)`. Otherwise it will be the normal size
of `x`.
"""
function matlabsize(x)
    sz = size(x)
    if length(sz)==0
        return 1,1
    elseif length(sz)==1
        return sz[1],1
    else
        return sz
    end
end

"""
     w = gaussm(a)
Fit a Gaussian distribution to dataset `a`.
"""
function gaussm(reg=0)
    params = Dict{String,Any}("reg"=>reg)
    return Prmapping("Gaussian density","untrained",fitGauss!,predictGauss,params,nothing)
end
function gaussm(a::Prdataset,reg=0)
    return a*gaussm(reg)
end
function fitGauss!(w::Prmapping, a::Prdataset)
    reg = w.data["reg"]
    X = +a
    dim = size(X,2)
    C = cov(X) + reg*I
    Z = 1/sqrt((2*pi)^dim * det(C))
    w.data["mean"] = mean(X,dims=1)
    w.data["icov"] = inv(C)
    w.data["Z"] = Z
    w.nrin = dim
    w.nrout = 1
    return w
end
function predictGauss(w::Prmapping, a::Prdataset)
    mn = w.data["mean"]
    iC = w.data["icov"]
    Z = w.data["Z"]
    X = +a
    dim = size(mn,2)
    da = X .- mn
    mahal = sum( (da*iC) .* da, dims=2)
    p = Z .* exp.(- mahal./ 2.0)
    out = deepcopy(a)
    out.data = p
    return out
end
# Parzen
"""
    w = parzenm(a,h=1)
Estimate a Parzen density with width parameter `h` on dataset `a`.
"""
function parzenm(h=1.0)
    params = Dict{String,Any}("h"=>h)
    return Prmapping("Parzen density","untrained",fitParzen!,predictParzen,params,nothing)
end
function parzenm(a::Prdataset, h=1.0)
    return a*parzenm(h)
end
function fitParzen!(w, a)
    w.data["X"] = +a
    w.nrin = size(a,2)
    w.nrout = 1
    return w
end
function predictParzen(w, a)
    # unpack
    h = w.data["h"]
    X = w.data["X"]
    # compute
    Z = -1.0 ./(h*h)
    D = sqeucldistm(+a, X)
    pred = mean(exp.(Z * D),dims=2)
    return pred
end

"""
    w = knnm(a,k=1)
Estimate a k-Nearest Neighbor density (with `k` neighbors) on dataset `a`.
Note that the density is not normalised; it is basically just p = 1/dist-to-kNN.
"""
function knnm(k::Int=1)
    params = Dict{String,Any}("k"=>k)
    return Prmapping("kNN density","untrained",fitKNNm!,predictKNNm,params,nothing)
end
function knnm(a::Prdataset,k=1)
    return a*knnm(k)
end
function fitKNNm!(w,a)
    if w.data["k"]>size(a,1)
        error("More neighbors requested than available in the dataset.")
    end
    # the only thing we can do is storing the training data
    w.data["traindata"] = +a
    w.nrin = size(a,2)
    w.nrout = 1
    return w
end
function predictKNNm(w,a)
    k = w.data["k"]
    X = w.data["traindata"]
    D = sort(sqeucldistm(+a,X),dims=2)
    out = 1.0 ./D[:,[k]]
    return out
end

"""
       w = bayesc(a)
       p = a*w
Apply Bayes rule to the probabilities stored in `a`. The output of the
mapping `w` is a  dataset `p` contains the posterior probabilities for
the classes (where the class labels are stored in `b.featlab`.
"""
function bayesc()
    params = Dict{String,Any}()
    return Prmapping("Bayes","untrained",fitBayes!,predictBayes,params,nothing)
end
function bayesc(a::Prdataset)
    return a*bayesc()
end
function fitBayes!(w::Prmapping, a::Prdataset)
    prior = classpriors(a)
    prior = reshape(prior,1,length(prior)) # make a row vector
    w.data["prior"] = prior
    w.labels = a.lablist
    w.nrin = size(a,2)
    w.nrout = length(prior)
    return w
end
function predictBayes(w::Prmapping, a::Prdataset)
    prior = w.data["prior"]
    if (size(a,2) != length(prior))
        error("Number of priors does not correspond to the number of class.cond.P")
    end
    if any(+a .< 0.0)
        error("Dataset should contain probabilities (so only positive values).")
    end
    Ppost = +a .* prior
    Ppost = Ppost ./ sum(Ppost,dims=2)
    out = deepcopy(a)
    out.data = Ppost
    out.featlab = w.labels
    return out
end

"""
        w =  generativem(a,dens=gaussm)
Fit a density mapping `dens` to each of the classes in `a`. When for
each of the classes need another density, a vector of untrained density
mappings can be supplied in `dens`.
"""
function generativem(dens=gaussm)
    params = Dict{String,Any}("dens"=>dens)
    return Prmapping("Generative map","untrained",fitGenerative!,predictGenerative,params,nothing)
end
function generativem(a::Prdataset, dens)
    return a*generativem(dens)
end
function generativec(dens=gaussm)
    params = Dict{String,Any}("dens"=>dens)
    u1 = Prmapping("Generative cl.","untrained",fitGenerative!,predictGenerative,params,nothing)
    u2 = bayesc()
    u = u1*u2
    u.name = "Generative cl."
    return u
end
function fitGenerative!(w::Prmapping, a::Prdataset)
    dens = w.data["dens"]
    c = nrclasses(a)
    # Define all density estimators for each of the classes:
    if (dens isa Prmapping)
        u = Vector{Prmapping}(undef,c)
        for i=1:c
            u[i] = deepcopy(dens)
        end
    else
        if (length(dens) != c)
            error("Number of densities does not fit the nr of classes.")
        end
        u = dens
    end
    # Train all the density estimators:
    I = findclasses(a)
    v = Vector{Prmapping}(undef,c)
    for i=1:c
        v[i] = a[I[i],:] * u[i]
    end
    # Store everything
    w.data["Pclass"] = v
    w.data["priors"] = classpriors(a)
    w.labels = a.lablist
    w.nrin = size(a,2)
    w.nrout = c
    return w
end
function predictGenerative(w::Prmapping, a::Prdataset)
    # unpack
    Pclass = w.data["Pclass"]
    prior = w.data["priors"]
    # go
    c = length(prior)
    n = size(a,1)
    P = zeros(n,c)
    for i=1:c
        P[:,i] = +( out = a*Pclass[i] )
    end
    # Store it:
    out = deepcopy(a)
    out.data = P
    out.featlab = w.labels
    return out
end

# Some simple classifiers:
"""
    w = qdc(a,reg=0.0)
Fit a Quadratic Discriminant classifier on dataset `a`.
"""
function qdc(reg=0.0)
    out = generativec(gaussm(reg))
    out.name = "Quadratic discr."
    return out
end
function qdc(a::Prdataset,reg=0.0)
    return a*qdc(reg)
end
"""
    w = parzenc(a,h=1)
Fit a Parzen classifier, with width parameter `h`, on dataset `a`.
"""
function parzenc(h=1.0)
    u = generativec(parzenm(h))
    u.name = "Parzen classifier"
    return u
end
function parzenc(a::Prdataset,h=1.0)
    return a*parzenc(h)
end
"""
    w = knnc(a,k=1)
Fit a k-Nearest Neighbor classifier, with the number of neighbors `k`,
on dataset `a`.
"""
function knnc(k::Int=1)
    params = Dict{String,Any}("k"=>k)
    return Prmapping("kNN classifier","untrained",fitKNNc!,predictKNNc,params)
end
function knnc(a::Prdataset,k=1)
    return a*knnc(k)
end
function fitKNNc!(w,a)
    # the only thing we can do is storing the training data
    w.data["traindata"] = a
    w.nrin = size(a,2)
    w.nrout = nrclasses(a)
    w.labels = a.lablist
    return w
end
function predictKNNc(w,a)
    k = w.data["k"]
    X = w.data["traindata"]
    y = X.nlab
    n = size(a,1)
    C = w.nrout
    pred = zeros(n,C)
    D = sqeucldistm(+a,+X)
    # Ooph, do we really need a loop here?
    for i=1:n
       J = sortperm(D[i,:])  # sort all distances
       lab = y[J[1:k]]       # find the labels of the k nearest objects
       pred[i,:] .= proportions(lab,C)
    end
    return pred
end

