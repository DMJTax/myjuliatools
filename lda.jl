using Statistics

export ldc

# Fit the parameters of an LDA:
# 0. (class priors?)
# 1. the means
# 2. the average covariance matrix
function fitLDA!(w, a)
   lambda = w.data["lambda"]
   dim = size(a,2)
   c = nrclasses(a)
   I = findclasses(a)
   X = +a
   means = Matrix{AbstractFloat}(undef,c,dim)
   covs = Array{AbstractFloat}(undef,dim,dim,c)
   for i=1:c
      means[i,:] = mean(X[I[i],:],dims=1)
      covs[:,:,i] = cov(X[I[i],:],dims=1)
   end
   C = dropdims(mean(covs,dims=3), dims=3)
   w.data["means"] = means
   w.data["invcov"] = inv(C)
   w.data["priors"] = classpriors(a)
   w.labels = a.lablist
   w.nrin = dim
   w.nrout = c
   return w
end
function predictLDA(w, a)
   # unpack
   means = w.data["means"]
   invC = w.data["invcov"]
   priors = w.data["priors"]
   # compute
   c = size(means,1)
   N = size(a,1)
   pred = zeros(N,c)
   for i=1:c
      df = a.data .- means[[i],:]
      dist = (df*invC) .* df 
      pred[:,i] = exp.(- sum(dist,dims=2))
   end
   # store it:
   out = deepcopy(a)
   out.data = pred
   out.featlab = w.labels # don't forget the corresponding class labels
   return out
end
"""
    w = ldc(a)
Fit a Linear Discriminant Analysis classifier on dataset `a`.
"""
function ldc(lambda=0.0)
   params = Dict{String,Any}("lambda"=>lambda)
   return Prmapping("Linear Discriminant Classifier","untrained",fitLDA!,predictLDA,params,nothing)
end
function ldc(a::Prdataset,lambda)
   return a*ldc(lambda)
end

