# Define some useful datasets, for regression and classification
#
export gendats,gendatb,gendatsin,gendatsimple2D,gendatparab,gendatsinc,highdimr

"""
    a = gendats(n=[50 50],d=1,dim=2)
Simple classification problem with 2 Gaussian classes, with distance `d`.
"""
function gendats(n=[50 50],d=1,dim=2)
    n = genclass(n, [0.5 0.5])
    x1 = randn(n[1],dim) 
    delta = zeros(1,dim)
    delta[1] = d
    x2 = randn(n[2],dim) .+ delta
    out = Prdataset([x1;x2],genlab(n,["ω_1" "ω_2"]),"Simple dataset")
    out.featlab = ["Feature 1", "Feature 2"]
    if (dim>2)
        out.data = [out.data randn(sum(n),dim-2)]
    end
    return out
end

"""
    a = gendatb(n=[50 50],s=1)
Simple banana-shaped classification problem with 2 Gaussian classes, with spread `s`.
"""
function gendatb(n=[50 50], s=1.0)
    R = 5
    n = genclass(n,[0.5 0.5])

    domaina = 0.125*π .+ rand(n[1],1)*1.25*π;
    a = [R.*sin.(domaina) R.*cos.(domaina)] .+ randn(n[1],2).*s;

    domainb = 0.375*π .- rand(n[2],1)*1.25*π;
    a = [a; [R.*sin.(domainb) R.*cos.(domainb)] .+ randn(n[2],2)*s .+ 
           [-0.75*R -0.75*R]];
    #ones(N(2),1)*[-0.75*r -0.75*r]];
    lab = genlab(n,["apple"; "pear"]);
    out = Prdataset(a,lab,"Banana dataset")
    out.featlab = ["Feature 1", "Feature 2"]
    return out
end

"""
    a = gendatsin(n=40,s=0.1)
Simple 1D sinusoidal regresson problem, with spread `s`.
"""
function gendatsin(n=40,s=0.1)
    x = π * (2*rand(n,1) .- 1.0)
    y = sin.(x) .+ s*randn(n,1)
    return Prdataset(x,y,"Sinusoidal dataset")
end
"""
    a = gendatsimple2D(n=50)
Simple 2D sinusoidal regresson problem.
"""
function gendatsimple2D(n=40,s=0.0)
    x = randn(n,2)
    y = sin.(x[:,1]) .+ 0.1*x[:,2] .+ s*randn(n,1)
    return Prdataset(x,y,"Simple 2D dataset")
end
"""
    a = gendatparab(n=40,s=0.1)
Simple 1D parabolic regresson problem, with noise standard deviation `s`.
"""
function gendatparab(n=40,s=0.1)
    x = 2*rand(n,1) .- 1.0
    y = 1.0 .- x.*x .+ s*randn(n,1)
    return Prdataset(x,y,"Parabolic dataset")
end
"""
    a = gendatsinc(n=40,s=0.1)
Simple 1D sinc regresson problem, with noise standard deviation `s`.
"""
function gendatsinc(n=40,s=0.1)
    x = 3*π * (2*rand(n,1) .- 1.0)
    y = sin.(x)./x .+ s*randn(n,1)
    return Prdataset(x,y,"Sinc dataset")
end

"""
       highdimr(n,dim,σ=0.1)

Generate `n` samples from a high-dimensional data of `dim` dimensions, where the input is drawn from a standard Gaussian distribution, and the output is defined as:
```
     y = sin([1,1,..,1] x) + ε
```
where ε is drawn from a Gaussian with standard deviation `σ`.
"""
function highdimr(n,dim,sigma=0.1)
    x = randn(n,dim)
    w = ones(dim)
    y = sin.(x*w) .+ sigma*randn(n,1)
    return Prdataset(x,y,"High-dim sinusoidal")
end


