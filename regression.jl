"""
Regression

Implement the following regression models:

kNNr            k-Nearest Neighbor Regression
ksmoothr        Kernel smoother
locallinearr    Local Linear Regression
kridger         Kernel ridge Regression
gpr             Gaussian Process Regression
linearsmoother  Linear Smoother Regression

Helper functions:
PeffSmoother    Effective number of parameters of a smoother 
plotgr!         Plot output of Gaussian Process, including confidence bounds
"""

using LinearAlgebra

export kNNr, ksmoothr, locallinearr, kridger, gpr, linearsmoother, PeffSmoother, plotgpr!

"""
      w = kNNr(a,k=1)
Fit a k-nearest neighbor smoother on dataset `a`, using `k` nearest neighbors.
"""
function kNNr(k=1)
    # This is the untrained mapping
    params = Dict{String,Any}("k"=>k)
    return Prmapping("k-Nearest Neighbor Regression","untrained",fitKNNr!,predictKNNr,params,nothing)
end
# The call directly with some dataset:
function kNNr(a::Prdataset,k=1)
    return a*kNNr(k)
end
function fitKNNr!(w,a)
    w.data["traindata"] = a
    w.nrin = size(a,2)
    w.nrout = 1
    return w
end
function predictKNNr(w,a)
    k = w.data["k"]
    x = w.data["traindata"]
    n = size(a,1)
    out = zeros(n,1)
    for i=1:n
        ai = +a[[i],:]
        D = sqeucldistm(ai,+x)
        I = sortperm(D,dims=2)
        out[[i],1] .= mean(x.targets[I[1:k]]) 
    end
    return out
end


"""
      w =  ksmoothr(a, σ=1.0)
Fit a Kernel smoother on dataset `a`, using a Gaussian kernel, with standard deviation of `σ`.
"""
function ksmoothr(sigma=1.0)
    # This is the untrained mapping
    params = Dict{String,Any}("sigma"=>sigma)
    return Prmapping("Kernel Smoother Regression","untrained",fitKSmooth!,predictKSmooth,params,nothing)
end
# The call directly with some dataset:
function ksmoothr(a::Prdataset,sigma=1.0)
    return a*ksmoothr(sigma)
end
function fitKSmooth!(w,a)
    w.data["traindata"] = a
    w.nrin = size(a,2)
    w.nrout = 1
    return w
end
function predictKSmooth(w,a)
    sigma = w.data["sigma"]
    x = w.data["traindata"]
    n = size(a,1)
    out = zeros(n,1)
    for i=1:n
        ai = +a[[i],:]
        k = gaussianKernel(ai,x,sigma)
        out[[i],1] = k*x.targets / sum(k)
    end
    return out
end

"""
    w = locallinearr(a,σ=1.0,λ=0.001)
Fit a Local Linear Regression on dataset `a`, where a Gaussian kernel with width `σ` is used.
"""
function locallinearr(sigma=1.0,lambda=0.001)
    # This is the untrained mapping
    params = Dict{String,Any}("sigma"=>sigma,"lambda"=>lambda)
    return Prmapping("Local Linear Regression","untrained",fitLocalLinear!,predictLocalLinear,params,nothing)
end
# The call directly with some dataset:
function locallinearr(a::Prdataset,sigma=1.0,lambda=0.001)
    return a*locallinearr(sigma,lambda)
end

function fitLocalLinear!(w,a)
    w.data["traindata"] = a
    w.nrin = size(a,2)
    w.nrout = 1
    return w
end
function predictLocalLinear(w,a)
    sigma = w.data["sigma"]
    lambda = w.data["lambda"]
    x = w.data["traindata"]
    X1 = [ones(size(x,1),1) +x]
    n = size(a,1)
    out = zeros(n,1)
    for i=1:n
        ai = +a[[i],:]
        k = gaussianKernel(ai,x,sigma)
        Cinv = inv(X1' * (k' .* X1) + I*lambda)
        out[[i],1] = [1.0 ai]*Cinv*X1'*(vec(k).*x.targets)
    end
    return out
end

"""
      w =  kridger(a,σ=1,λ=0.01)
Train a kernelized ridge regression on dataset `a`. The kernel is a
Gaussian kernel with width `σ`, and the regularization parameter for the
L2 loss is `λ`.
"""
function kridger(sigma=1.0,lambda=0.01)
    # This is the untrained mapping
    params = Dict{String,Any}("sigma"=>sigma,"lambda"=>lambda)
    return Prmapping("Kernel Ridge Regression","untrained",fitKridge!,predictKridge,params,nothing)
end
# The call directly with some dataset:
function kridger(a::Prdataset,sigma=1.0,lambda=0.01)
    return a*kridger(sigma,lambda)
end
function fitKridge!(w,a)
    sigma = w.data["sigma"]
    lambda = w.data["lambda"]
    K = gaussianKernel(+a,+a,sigma) + I*lambda
    Cinv = inv(K)
    weights = a.targets' * Cinv
    w.data["weights"] = weights
    w.data["traindata"] = a
    w.data["Cinv"] = Cinv
    w.nrin = size(a,2)
    w.nrout = 1
    return w
end
function predictKridge(w,a)
    sigma = w.data["sigma"]
    weights = w.data["weights"]
    x = w.data["traindata"]
    n = size(a,1)
    out = zeros(n,1)
    for i=1:n
        ai = +a[[i],:]
        k = gaussianKernel(ai,x,sigma)
        out[[i],1] = weights * k'
    end
    return out
end


"""
    w = gpr(a,σ_k=1.0,β=0.1,w_0=1.0,outputsigma=true)

Fit a Gaussian Process Regression on dataset `a`. Parameter `σ_k`
determines the scaling factor in the Gaussian kernel, and `β` is the
noise variance, and `w_0` is the variance of the prior distribution over
the weights.
""" 
function gpr(sigma=1.0,beta=0.1,w0=1.0,outputsigma=true)
    # This is the untrained mapping
    params = Dict{String,Any}("sigma"=>sigma, "beta"=>beta,"w0"=>w0,"outputsigma"=>outputsigma)
    return Prmapping("Gaussian Process Regression","untrained",fitGP!,predictGP,params,nothing)
end
# The call directly with some dataset:
function gpr(a::Prdataset,sigma=1.0, beta=0.1,w0=1.0,outputsigma=true)
    return a*gpr(sigma,beta,w0,outputsigma)
end

function fitGP!(w,a)
    sigma = w.data["sigma"]
    beta = w.data["beta"]
    w0 = w.data["w0"]
    K = w0*gaussianKernel(a,a,sigma) + I*beta
    w.data["Cinv"] = inv(K)
    w.data["traindata"] = a
    w.nrin = size(a,2)
    if w.data["outputsigma"]
        w.nrout = 2
    else
        w.nrout = 1
    end
    return w
end
function predictGP(w,a)
    sigma = w.data["sigma"]
    Cinv = w.data["Cinv"]
    outputsigma = w.data["outputsigma"]
    w0 = w.data["w0"]
    X = w.data["traindata"]
    n = size(a,1)
    if outputsigma
        out = zeros(n,2)
        featlab = ["mu","std"]
    else
        out = zeros(n,1)
        featlab = ["mu"]
    end
    for i=1:n
        ai = +a[[i],:]
        c = w0*gaussianKernel(ai,ai,sigma)
        k = w0*gaussianKernel(ai,X,sigma)
        out[[i],1] = k*Cinv*X.targets
        if outputsigma
            out[[i],2] = c - k*Cinv*k'
        end
    end
    b = deepcopy(a)
    b.data = out
    b.featlab = featlab
    return b
end
function plotgpr!(w::Prmapping,gridsize=100,color=:black)
    xl = xlims()
    #z = collect(xl[1]:0.1:xl[2])
    z = range(xl[1],xl[2],length=gridsize)
    out = Prdataset(z[:,:])*w
    twostd = 2.0*+out[:,2]
    plot!(z,+out[:,1],color=color)
    plot!(z,+out[:,1].+twostd,color=color,linestyle=:dash,seriestype=:path,label="")
    plot!(z,+out[:,1].-twostd,color=color,linestyle=:dash,seriestype=:path,label="")
end

"""
    w = linearsmoother(a, reg=0, usebias=true)

Fit a linear least square regression on dataset `a`, possibly
regularized by `reg`. Per default a bias term is included, but if that
is not needed, then use `usebias=false`. 
This regressor is the same as `linearr`, except that it also stores the
Smoothing matrix (needed for the computation of the effective number of
parameters). If you don't need the effective number of parameters, it is
more efficient to use `linearr`.
"""

function linearsmoother(reg=0.0, usebias=true)
    # This is the untrained mapping
    params = Dict{String,Any}("reg"=>reg, "usebias"=>usebias)
    return Prmapping("Linear Smoother regression","untrained",fitLinearSmoother!,predictLinearSmoother,params,nothing)
end
# The call directly with some dataset:
function linearsmoother(a::Prdataset,reg=0.0, usebias=true)
    return a*linearsmoother(reg,usebias)
end
function fitLinearSmoother!(w, a)
    # Unpack the parameters
    reg = w.data["reg"] # ungainly storage
    usebias = w.data["usebias"]
    # Do the work
    if usebias
        n = size(a.data,1)
        X = [ones(n,1) a.data]
    end
    if (reg>0.0)
        dim = size(X,2)
        S = inv(X'*X .+ reg*I(dim))*X'
    else
        S = inv(X'*X)*X'
    end
    beta = S*a.targets
    # Store the results
    w.data["weights"] = beta
    w.data["S"] = S
    w.data["traindata"] = a.targets # not really needed, but to keep the implementation of `PeffSmoother` simple, include a matrix with n rows. 
    w.nrin = size(a,2)
    w.nrout = size(a.targets,2)
    return w
end
function predictLinearSmoother(w, a)
    # Unpack the parameters
    usebias = w.data["usebias"] # ungainly storage
    w = w.data["weights"] # ungainly storage
    # Do the work
    if usebias
        n = size(a.data,1)
        X = [ones(n,1) a.data]
    end
    out = X*w
    return out
end


"""
   p = PeffSmoother(w)
Compute the number of effective parameters for a trained smoother `w`.
It is defined for `ksmoothr`, `locallinearr`, `gpr`.
"""
function PeffSmoother(w)
    x = w.data["traindata"]
    n = size(x,1)
    shat = zeros(n,1)
    if (w.name=="Linear Smoother regression")
        S = w.data["S"]
        peff = diag(S)'*diag(S)
    elseif (w.name=="Kernel Smoother Regression")
        sigma = w.data["sigma"]
        for i=1:n
            xi = +x[[i],:]
            k = gaussianKernel(xi,x,sigma)
            s = k/sum(k)
            shat[[i],1] = s*s'
        end
        peff = sum(shat)
    elseif (w.name=="Local Linear Regression")
        sigma = w.data["sigma"]
        lambda = w.data["lambda"]
        X1 = [ones(size(x,1),1) +x]
        for i=1:n
            xi = +x[[i],:]
            k = gaussianKernel(xi,x,sigma)
            Cinv = inv(X1' * (k' .* X1) + I*lambda)
            s = [1.0 xi]*Cinv*X1'.*k
            shat[[i],1] = s*s'
        end
        peff = sum(shat)
    elseif (w.name=="Gaussian Process Regression") | (w.name=="Kernel Ridge Regression")
        sigma = w.data["sigma"]
        Cinv = w.data["Cinv"]
        for i=1:n
            xi = +x[[i],:]
            k = gaussianKernel(xi,x,sigma)
            s = k*Cinv
            shat[[i],1] = s*s'
        end
        peff = sum(shat)
    else
        warning("For ",w.name," no effective number of parameters has been defined.")
    end
    return peff
end

