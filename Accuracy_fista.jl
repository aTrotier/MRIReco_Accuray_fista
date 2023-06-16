using MRIReco, MRIFiles, MRICoilSensitivities, MRIOperators
using RegularizedLeastSquares: Regularization, createLinearSolver, solve
using BartIO
using CairoMakie
using ImageQualityIndexes:assess_ssim

set_bart_path("/usr/local/bin/bart")

T = Float32

################################
# build acquisition data
################################
## bart part
phantom = bart(1,"phantom");
#phantom = bart(1,"noise -n0.005",phantom)
N = 128
NCh = 8
phant3D = bart(1,"phantom -3 -x$N -s$NCh");
#phant3D = bart(1,"noise -n0.005",phant3D)
phant3D_rss = bart(1,"rss 8",phant3D)
kbart = bart(1,"fft -u 7",phant3D);
mask = bart(1,"poisson -Y $N -Z $N -y1.2 -z1.2 -C 20 -v -V5");
kbart_u = kbart .* mask;

# MRIReco fully
acqFully = AcquisitionData(reshape(kbart,size(kbart_u)...,1,1)) # from bart ?

# MRIReco CS
tr = MRIBase.CartesianTrajectory3D(T, N, N, numSlices=N, TE=T(0), AQ=T(0))
kdata_j = [reshape(Complex{T}.(kbart),:,NCh) for i=1:1, j=1:1, k=1:1]
acq = AcquisitionData(tr, kdata_j, encodingSize=(N,N,N))

I = findall(x->x==1,abs.(repeat(mask,N,1,1)))
subsampleInd = LinearIndices((N,N,N))[I]
acqCS = deepcopy(acq);
acqCS.subsampleIndices[1] = subsampleInd
acqCS.kdata[1,1,1] = acqCS.kdata[1,1,1][subsampleInd,:]


# MRIReco low level
kdata= multiCoilData(acqCS, 1, 1, rep=1) #/ Float32(192935.546875) # if we want the same inverse scaling



################################
# generate coil sensitivity maps
################################
sensitivity =  bart(1,"ecalib -m1 -c0", kbart_u);

################################
# build Low level functions and operators
################################

reconSize = acqCS.encodingSize

#E = encodingOps_parallel(acqCS, reconSize, params[:senseMaps]; slice=1)[1] # high level
idx = acqCS.subsampleIndices[1]
sampOp = SamplingOp(idx, reconSize, Complex{T})
ftOp = FFTOp(Complex{T}, reconSize; unitary=false)
S = SensitivityOp(reshape(Complex{T}.(sensitivity),:,NCh),1)
E = DiagOp(sampOp ∘ ftOp, NCh) ∘ S

EᴴE = normalOperator(E)

λ = Float32.(0.01)
reg = Regularization("L1", λ; shape=reconSize)
sparseTrafo = SparseOp(Complex{T},"Wavelet", reconSize)

################################
# high level parameters
################################
params = Dict{Symbol, Any}()
params[:reco] = "multiCoil"
params[:senseMaps] = Complex{T}.(sensitivity);

params[:solver] = "fista"
params[:sparseTrafoName] = "Wavelet"
params[:regularization] = "L1"
params[:λ] = T(0.01) # 5.e-2
params[:normalize_ρ] = true
params[:ρ] = T(0.95)
#params[:relTol] = 0.1
params[:normalizeReg] = true

################################
# reconstruction fully 
################################

# BART
imFully_bart = bart(1,"pics -d5 -i1 l2 -r0",kbart,sensitivity);

# julia high levelparams = Dict{Symbol, Any}()
params2 = Dict{Symbol, Any}()
params2[:reco] = "multiCoil"
params2[:senseMaps] = Complex{T}.(sensitivity);

params2[:solver] = "cgnr"
params2[:regularization] = "L2"
params2[:λ] = T(0.01)
params2[:iterations] = 1

imFully_julia = reconstruction(acqFully, params).data[:,:,:,1,1,1];

# julia low level


EF = DiagOp( ftOp, NCh) ∘ S
EFᴴEF = normalOperator(EF)

solverFully = createLinearSolver("cgnr", EF; 
AᴴA=EFᴴEF, 
reg=Regularization("L2", 0.01),
relTol=Float64(eps(real(T))),
iterations=1,
normalizeReg=true,
verbose = false)

kdataFully = multiCoilData(acqFully, 1, 1, rep=1) #/ Float32(192935.546875) # if we want the same inverse scaling

imFully_julia_ll = solve(solverFully, kdataFully) 
imFully_julia_ll = reshape(imFully_julia_ll,reconSize)


################################
# reconstruction loop
################################
img_bart_vec = Vector{Array{ComplexF32,3}}()
img_julia_vec = Vector{Array{ComplexF32,3}}()
img_julia_ll_vec = Vector{Array{ComplexF32,3}}()

RMSE_bart = Vector{AbstractFloat}()
ssim_bart = Vector{AbstractFloat}()
RMSE_julia = Vector{AbstractFloat}()
ssim_julia = Vector{AbstractFloat}()
RMSE_julia_ll = Vector{AbstractFloat}()
ssim_julia_ll = Vector{AbstractFloat}()

iter_vec = [1,5,30,50,150]
for iter in iter_vec
  # bart reco
  imBART = bart(1,"pics -d5 -i $iter -RW:7:0:0.01", kbart_u, sensitivity)
  push!(img_bart_vec,imBART)
  push!(RMSE_bart,MRIReco.norm(vec(abs.(imBART))-vec(abs.(imFully_bart)))/MRIReco.norm(vec(abs.(imFully_bart))))
  push!(ssim_bart,round(assess_ssim(abs.(imBART[:,:,80]),abs.(imFully_bart[:,:,80])),digits=3))

  # high level reco
  params[:iterations] = iter
  imMRIReco = reconstruction(acqCS, params).data;
  push!(img_julia_vec,imMRIReco[:,:,:,1,1,1])
  push!(RMSE_julia,MRIReco.norm(vec(abs.(imMRIReco[:,:,:,1,1,1]))-vec(abs.(imFully_julia)))/MRIReco.norm(vec(abs.(imFully_julia))))
  push!(ssim_julia,round(assess_ssim(abs.(imMRIReco[:,:,80,1,1,1]),abs.(imFully_julia[:,:,80])),digits=3))

  # low level reco
  solver = createLinearSolver("fista", E; 
  AᴴA=EᴴE, 
  reg=reg,
  ρ=0.95, 
  normalize_ρ=true,
  t=1,
  relTol=eps(real(T)),
  iterations=iter,
  normalizeReg=true,
  restart = :none,
  verbose = false)


  I = solve(solver, kdata) 
  img_julia_ll = reshape(I,reconSize)
  push!(img_julia_ll_vec,img_julia_ll)
  push!(RMSE_julia_ll,MRIReco.norm(vec(abs.(img_julia_ll))-vec(abs.(imFully_julia_ll)))/MRIReco.norm(vec(abs.(imFully_julia_ll))))
  push!(ssim_julia_ll,round(assess_ssim(abs.(img_julia_ll[:,:,80]),abs.(imFully_julia_ll[:,:,80])),digits=3))
end

begin
  f=Figure()
  ax = Axis(f[1,1],title = "RMSE")
  lines!(iter_vec,RMSE_bart,label="BART")
  lines!(iter_vec,RMSE_julia,label="julia HL")
  lines!(iter_vec,RMSE_julia_ll,label="julia LL")

  ax = Axis(f[1,2],title = "SSIM")
  lines!(iter_vec,ssim_bart,label="BART")
  lines!(iter_vec,ssim_julia,label="julia HL")
  lines!(iter_vec,ssim_julia_ll,label="julia LL")
  Legend(f[1,3],ax)
  f
end
save("plots/compare_metrics.pdf",f)


## plots images
begin
slice = 80

f = Figure(resolution=(400,500))
ga = f[1,1] = GridLayout()
asp = 1
for i in 1:length(img_julia_vec)
  
  ax1 = Axis(ga[i,1],aspect=asp)
  hidedecorations!(ax1)
  heatmap!(ax1,abs.(img_julia_vec[i][:,:,slice]),colormap=:grays)
 

  ax2 = Axis(ga[i,2],aspect=asp)
  hidedecorations!(ax2)
  heatmap!(ax2,abs.(img_julia_ll_vec[i][:,:,slice]),colormap=:grays)

  ax3 = Axis(ga[i,3],aspect=asp)
  hidedecorations!(ax3)
  heatmap!(ax3,abs.(img_bart_vec[i][:,:,slice]),colormap=:grays)

  Label(ga[i,0],"iter = $(iter_vec[i])",tellheight = false)

  if i == 1
    ax1.title = "MRIReco \n high level"
    ax2.title = "julia \n low level"
    ax3.title = "bart \n fista"
  end
  rowsize!(ga,i,75)
end
rowgap!(ga,0)
Label(ga[0,:],"")
f
end

save("plots/compare_img.pdf",f)