using Statistics
using LinearAlgebra

export pca

"""
    w = pca(a)
    w = pca(frac)

Fit a PCA on dataset `a`, retain `n` dimensions. If `frac` (between 0 and 1) is given, this fraction of variance is retained.
"""
function pca(n=2)
   params = Dict{String,Any}("n"=>n)
   return Prmapping("Principal Component Analysis","untrained",fitPCA!,predictPCA,params,nothing)
end
function pca(a::Prdataset,n)
   return a*pca(n)
end
# Fit the parameters of an PCA:
# 1. the mean
# 2. the covariance matrix
function fitPCA!(w, a)
   n = w.data["n"]
   dim = size(a,2)
   X = +a
   datamean = mean(X,dims=1)
   datacov = cov(X)

   lambda = eigvals(datacov)
   v = eigvecs(datacov)
   J = sortperm(lambda,rev=true)
   if (n<1.0) # we defined a fraction of variance explained
      f = cumsum(l[J])./sum(l)
      n = sum(f.<=n)
   end
   V = v[:,J[1:n]]
   # store the results
   w.data["mean"] = datamean
   w.data["pcs"] = V
   return w
end
function predictPCA(w, a)
   # unpack
   mean = w.data["mean"]
   pcs = w.data["pcs"]
   # compute
   df = a.data .- mean
   pred = df*pcs
   # store it:
   out = deepcopy(a)
   out.data = pred
   return out
end


