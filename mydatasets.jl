# Define some useful datasets, for regression and classification
#
export gendats,gendatb,gendatsin

"""
    a = gendats(n=[50 50],d=1,dim=2)
Simple classification problem with 2 Gaussian classes, with distance `d`.
"""
function gendats(n=[50 50],d=1,dim=2)
    n = genclass(n, [0.5 0.5])
    x1 = randn(n[1],2) 
    x2 = randn(n[2],2) .+ [d 0]
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
    lab = genlab(n);
    out = Prdataset(a,lab,"Banana dataset")
    out.featlab = ["Feature 1", "Feature 2"]
    return out
end

"""
    a = gendatsin(n=[50 50],s=1)
Simple 1D sinusoidal regresson problem, with spread `s`.
"""
function gendatsin(n=40,s=0.1)
    x = π * (2*rand(n,1) .- 1.0)
    y = sin.(x) .+ s*randn(n,1)
    return Prdataset(x,y,"Sinusoidal dataset")
end

