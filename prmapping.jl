# Try to define prmapping: what do we need to make it work?
#
# A prmapping can be in different 'states':
# - untrained:  parameters of the model are not defined yet
#               (data==nothing)
# - trained:    parameters of the model are defined
# - fixed:      parameters may be given, but the fit-function is not
#               defined (fit==nothing)
# simple least squares fitting
#
# To define a mapping, the user has to define 3 things:
# 1. an untrained mapping where hyperparameters can be defined/set:
#     u = linearr(degree=2)
# 2. a function that accepts a prdataset such that it can be trained:
#     w = fitf!(param, dataset)
#    It is expected that this fitting function is updating the
#    parameters param 
# 3. a function that processes a dataset to get a prediction out:
#     pred = predictf(param, dataset)
#
# An example is given by the linear regression `linearr`
#
# Next, mapping can be combined, for instance insequence:
# u = [pca() ldc()]
using Statistics
using Plots

export Prmapping,prmapping,plotr!,plotm!,plotc!,sequential

# Define a basic prmapping
# # DXD: should I make explicit somewhere if we're dealing with
# regression or classification mappings?
mutable struct Prmapping
    name          # name of the mapping
    type          # type of mapping (untrained/trained)
    fit!          # fit function
    predict       # predict function
    data          # parameters of the model
    labels        # feature labels of the output 
    nrin          # nr input features
    nrout         # nr of outputs
end
# Next we define an untrained mapping:
# (How should we store the parameters of the function: leave it to the
# user?)
function prmapping(fit::Function,predict::Function,data)
    return Prmapping(nothing,"untrained",fit,predict,data,nothing)
end
function prmapping(fit::Function,predict::Function)
    return Prmapping(nothing,"untrained",fit,predict,nothing,nothing)
end
function Prmapping(name::String,type::String,fit::Function,predict::Function,data,labels)
    return Prmapping(name,type,fit,predict,data,labels,0,0)
end
function Prmapping(name::String,type::String,fit::Function,predict::Function,data)
    return Prmapping(name,type,fit,predict,data,nothing,0,0)
end
# Show
function Base.show(io::IO, ::MIME"text/plain", w::Prmapping)
    if (w.name != nothing)
        print(w.name,", ")
    end
    if (w.nrin>0)
        print(w.nrin," to ",w.nrout," ")
    end
    print(w.type," mapping (")
    if (w.fit! !==nothing)
        print("fit: ",String(Symbol(w.fit!)),", ")
    end
    if (w.predict !==nothing)
        print("predict: ",String(Symbol(w.predict)))
    end
    print(")")
end

# The training step
function Base.:*(a::Prdataset,w::Prmapping)
    if w.type=="untrained"  # train an untrained mapping
        out = deepcopy(w)
        w.fit!(out, a) # DXD better?
        out.type = "trained"
        return out
    elseif w.type=="trained" # apply a trained mapping to data
        pred = w.predict(w,a)
        if (pred isa Prdataset)
            return pred
        else
            out = deepcopy(a)
            setfield!(out,:data,pred)
            return out
        end
    elseif w.type=="fixed"  # apply a fixed mapping to data
        out = deepcopy(a)
        setfield!(out,:data,w.predict(w,a))
        return out
    else
        error("This type of mapping is not defined yet.")
    end
end


# Plot a regression function
function plotr!(w::Prmapping)
    xl = xlims()
    z = collect(xl[1]:0.1:xl[2])
    out = Prdataset(z[:,:])*w
    plot!(z,+out)
end
# Plot a classification function
function plotm!(w::Prmapping, gridsize = 30)
    # dummy input to get to the nr of outputs:
    dummy = Prdataset([0.0 0.0])*w
    C = size(dummy,2)
    xl = xlims()
    yl = ylims()
    xrange = range(xl[1],xl[2],length=gridsize)
    yrange = range(yl[1],yl[2],length=gridsize)
    pred = zeros(gridsize,gridsize,C)
    for i=1:gridsize
        for j=1:gridsize
            input = Prdataset([xrange[i] yrange[j]])
            pred[i,j,:] = +(input*w)
        end
    end
    for i=1:C
        contour!(xrange,yrange,pred[:,:,i]')
    end
end
function plotc!(w::Prmapping, gridsize = 30)
    # dummy input to get to the nr of outputs:
    dummy = Prdataset([0.0 0.0])*w
    C = size(dummy,2)
    if C!=2
        error("Only two-class for now.")
    end
    xl = xlims()
    yl = ylims()
    xrange = range(xl[1],xl[2],length=gridsize)
    yrange = range(yl[1],yl[2],length=gridsize)
    pred = zeros(gridsize,gridsize,C)
    for i=1:gridsize
        for j=1:gridsize
            input = Prdataset([xrange[i] yrange[j]])
            pred[i,j,:] = +(input*w)
        end
    end
    df = pred[:,:,1]-pred[:,:,2]
    contour!(xrange,yrange,df[:,:,1]',levels=[0.0])
end

# Sequential mapping
function fitSeq!(w,a)
    u = w.data["mappings"]
    dim = size(a,2)
    K = length(u)
    trained = Vector{Prmapping}(undef,K)
    # train first mapping:
    if (u[1].type=="untrained")
        trained[1] = a*u[1]
    else
        trained[1] = deepcopy(u[1])
    end
    # map the data
    out = a*trained[1]
    # the rest
    for i=2:length(u)
        if (u[i].type=="untrained")
            trained[i] = a*u[i]
        else
            trained[i] = deepcopy(u[i])
        end
        out = out*trained[i]
    end
    c = size(out,2)
    w.data["mappings"] = trained
    w.labels = out.lablist
    w.nrin = dim
    w.nrout = c
    return w
end
function predictSeq(w,a)
    v = w.data["mappings"]
    # first mapping:
    out = a*v[1]
    for i=2:length(v)
        out = out*v[i]
    end
    return out
end
function sequential(u::Prmapping,u2...)
    K = length(u2)+1
    if K==1
        return u
    end
    mappings = Vector{Prmapping}(undef,K)
    mappings[1] = deepcopy(u)
    for i=2:K
        if !(u2[i-1] isa Prmapping)
            error("Only prmappings can be concatenated.")
        end
        mappings[i] = deepcopy(u2[i-1])
    end
    params = Dict{String,Any}("mappings"=>mappings)
    return Prmapping("Sequential map","untrained",fitSeq!,predictSeq,params,nothing)
end

# sequential mapping:
function Base.:*(u1::Prmapping,u2::Prmapping)
    return sequential(u1,u2)
end





