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

export sqeucldistm, gaussm, parzenm, qdc, parzenc, bayesc, generativem, generativec

"""
       D = sqeucldistm(A,B)
Compute the Squared Euclidean distance between the rows of matrix `A`
and the rows of matrix `B`. If `A` has size `NxD`, and `B` size `MxD`,
then output matrix is `NxM`.
"""
function sqeucldistm(A,B)
    n,dim = size(A)
    n2,dim2 = size(B)
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
Estimate a Parzen density on dataset `a`.
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
    u = [u1 u2]
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
    return generativec(gaussm(reg))
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


a = gendatb()
u = gaussm(0)
w = a*u
b = a*w
#scatterd(a)
#plotm!(w)

u = bayesc()
w = a*u

#u = generativem(gaussm())
#u = generativec(gaussm())
#u = generativec(parzenm(2.5))
u = qdc()
w = a*u
b = a*w


