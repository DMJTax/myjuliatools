using Random

export mogm,mogc

"""
   w = mogm(a,k) \
   w = mogm(a,k,ctype="full", reg=0.0001, nriters=100)

Fit a Mixture of Gaussians on dataset `a` with `k` components. The
components are fitted using EM, resulting in a set of `k` means,
covariance matrices and component priors. The output of the trained
mapping is for each of the samples of the dataset, the (unnormalised)
cluster probabilities.  

Finally, `ctype` defines the shape of the covariance matrices that you can impose on each of the Gaussian clusters:
ctype = "sphr"    spherical clusters
ctype = "full"    Gaussian clusters with a full covariance matrix per cluster
The covariance matrices are regularized by adding `reg` to the diagonals
of the covariance matrices.
Finally, the EM algorithm is run for *at max* `nriters` iterations.

"""
function mogm(k::Int=3,ctype="full",reg=0.0001,nriters::Int=100)
    params = Dict{String,Any}("k"=>k,"covtype"=>ctype,"nriters"=>nriters,"reg"=>reg)
    return Prmapping("Mixture of Gaussians","untrained",fitMOG!,predictMOG,params,nothing)
end
function mogm(a::Prdataset,k=3,ctype="full",reg=0.0001,nriters=100)
    return a*mogm(k,ctype,reg,nriters)
end
function fitMOG!(w,a)
    mog = deepcopy(w.data)
    k = w.data["k"]
    ctype = w.data["covtype"]
    nriters = w.data["nriters"]
    reg = w.data["reg"]
    mog["means"], mog["invcovs"], mog["priors"] = mog_init(+a,k,ctype)
    w.data = mog_update!(mog,+a,nriters,reg)
    w.nrin = size(a,2)
    w.nrout = 1
    return w
end
function predictMOG(w,a)
    covtype = w.data["covtype"]
    mn = w.data["means"]
    ic = w.data["invcovs"]
    pr = w.data["priors"]
    out = mog_P(+a,covtype,mn,ic,pr)
    return sum(out,dims=2)
end

# Initialise a mixture of gaussians
function mog_init(a,k,ctype)
    n,dim = size(a)
    if k>n
        error("More clusters requested than samples in dataset.")
    end
    I = randperm(n)
    mn = +a[I[1:k],:]
    if ctype=="sphr"
        icov = 1.0/mean( var(+a, dims=1) )
        icov = repeat([icov],k)
    elseif ctype=="full"
        icov = repeat(inv(cov(+a)),1,1,k)
    end
    pr = repeat([1/k], k)
    return mn,icov,pr
end
"""
          p = mog_P(x,ctype,means,invcovs,priors)

    Compute the density of data `x` under MoG characterized by the means
    `means`, (inverse) covariance matrices `invcovs` and priors
    `priors`. The output of the mapping is an NxK dataset, where for
    each of the `N` objects in `x` the `C` cluster probabilities (times
    the cluster prior) are returned.
"""
function mog_P(x,ctype,mn,icov,pr)
    n,dim = size(x)
    k = length(pr)
    D = sqeucldistm(x,mn)
    if ctype=="sphr"
        sig = 0.5*icov
        Z = (sig./pi) .^ (dim/2)
        P = Z' .* exp.(-sig'.*D)
    elseif ctype=="full"
        Z = (2*pi)^(-dim/2)
        P = zeros(n,k)
        for i=1:k
            df = x .- mn[[i],:]
            C = icov[:,:,i]
            P[:,i] .= sqrt(det(C)) * Z * exp.(-0.5*sum((df*C).*df,dims=2))
        end
    end
    return P.*pr' # include the prior
end

# Update the MoG (basically EM)
function mog_update!(mog,x,nriters=100,reg=0.0001)
    k = mog["k"]
    covtype = mog["covtype"]
    means = mog["means"]
    invcovs = mog["invcovs"]
    priors = mog["priors"]
    n,dim = size(x)

    iter = 1
    LL1 = -2e6    
    LL2 = -1e6    

    while (abs(LL2/LL1-1)>1e-6) && (iter<=nriters)
        println("Iteration $iter.")
        # calculate the old P:
        P = mog_P(x,covtype,means,invcovs,priors);

        # remember the LL:
        iter = iter+1;
        LL1 = LL2;
        sumP = sum(P,dims=2)
        LL2 = sum(log.(sumP))

        # normalize P:
        sumP[sumP.==0] .= 1;  # avoid division by 0
        normP = P./sumP

        # update the params:
        new_priors = sum(normP,dims=1);
        J = findall(new_priors.==0);
        new_priors[J] .= floatmin()*10; # avoid disappearing clusters
        new_means = normP' * x;

        # thus:
        priors = new_priors[:]/n;
        # normalize the means
        means = new_means./new_priors';
        # update inverse cov. matrices
        if covtype=="sphr"
            D = sqeucldistm(x,means);
            news = zeros(k,1);
            for i=1:k
                news[i] = (normP[:,i]'*D[:,i]);
            end
            covs = (news./new_priors[:])/dim .+ reg.*ones(k,1);
            invcovs = 1.0 ./covs;
        elseif covtype=="full"
            #println("update full cov.matrices")
            for i=1:k
                df = (x .- means[[i],:]) .* sqrt.(normP[:,i])
                invcovs[:,:,i] .= inv((df'*df)./new_priors[i] + reg*I) 
            end
        end
    end
    mog["means"] = means
    mog["invcovs"] = invcovs
    mog["priors"] = priors

    return mog
end
"""
     w = mogc(a,k)
     w = mogc(a,k,ctype="full",reg=0.0001,nriters=100)

Train a Mixture of Gaussians classifier on dataset `a`, using `k` clusters.
If requested, different types of clusters can be assumed:
  `ctype=sphr`  assumes spherical clusters
  `ctype=full`  assumes fully flexible covariance matrices
The covariance matrices are regularized by adding `reg` to the diagonals
of the covariance matrices.
Finally, the EM algorithm is run for *at max* `nriters` iterations.

"""
function mogc(k=3,ctype="full",reg=0.0001,nriters=100)
    out = generativec(mogm(k,ctype,reg,nriters))
    out.name = "MoG classifier"
    return out
end
function mogc(a::Prdataset,k=3,ctype="full",reg=0.0001,nriters=100)
    return a*mogc(k,ctype,reg,nriters)
end

