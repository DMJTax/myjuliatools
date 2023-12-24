# Try to test the prmapping
#

using Test
using myjuliatools


# test data generation:
N = (20,3)
X = randn(N)
# unlabeled data:
a = Prdataset(X)
@test isunlabeled(a)
@test size(a)==N
@test +a isa Matrix
# regression data
a = Prdataset(X,randn(N[1]))
@test isregression(a)
@test a.targets isa Vector


# Provided regression dataset 'gendatsin':
N = 27
a = gendatsin(N)
@test isregression(a)
@test !isclassification(a)
@test size(a)==(N,1)
c = [a; a]
@test size(c)==(2*N,1)
@test a.targets = randn(3,2) broken=true
a.targets = randn(N)
@test size(a,1)==N

# Provided classification dataset 'gendats':
N = 27
a = gendats(N)
@test !isregression(a)
@test isclassification(a)
@test size(a)==(N,2)
lab = genlab([3 4],["apple" "pear"])
@test a.labels = lab broken=true
b = gendats([3 7])
@test classsizes(b)==[3,7]

# Classification routines
b = Prdataset(repeat([1.0 0],10,1), genlab([6 4]))
nlab,ll = renumlab(getlabels(b))
b.featlab = ll
@test testc(b)==0.4



