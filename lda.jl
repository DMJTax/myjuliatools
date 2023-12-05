using LinearAlgebra

export fitLDA, predictLDA

# Define the Linear Discriminant Analysis classifier
mutable struct LDAc
   mean
   cov_inv
   threshold
end

"""
      fitLDA(x,λ=1e-6)

"""
function fitLDA(x,λ=1e-6)
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

