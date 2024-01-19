# -*- coding: utf-8 -*-
using Pkg
using Optim
using Roots
using NPZ
using Distributed
using CSV
using Tables
using ArgParse

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--gamma"
            help = "gamma"
            arg_type = Float64
            default = 8.0
        "--rho"
            help = "rho"
            arg_type = Float64
            default = 1.00001   
        "--Delta"
            help = "Delta"
            arg_type = Float64
            default = 1000.  
        "--dataname"
            help = "dataname"
            arg_type = String
            default = "output"
        "--zscale"
            help = "zscale"
            arg_type = Float64
            default = 1.0
        "--foc"
            help = "foc"
            arg_type = Int
            default = 1
        "--clowerlim"
            help = "clowerlim"
            arg_type = Float64
            default = 0.0001
        "--delta"
            help = "delta"
            arg_type = Float64
            default = 0.002
        "--zmax"
            help = "zmax"
            arg_type = Float64
            default = 2.0
        "--q"
            help = "q"
            arg_type = Float64
            default = 0.05
        "--distorted"
            help = "distorted"
            arg_type = Int
            default = 0
        "--boundary"
            help = "boundary"
            arg_type = Int
            default = 0
        "--logadjustment"
            help = "logadjustment"
            arg_type = Int
            default = 0
    end
    return parse_args(s)
end

@show parsed_args = parse_commandline()
gamma                = parsed_args["gamma"]
rho                  = parsed_args["rho"]
Delta                = parsed_args["Delta"]
dataname             = parsed_args["dataname"]
zscale                = parsed_args["zscale"]
foc                   = parsed_args["foc"]
clowerlim             = parsed_args["clowerlim"]
delta                 = parsed_args["delta"]
q                     = parsed_args["q"]
distorted             = parsed_args["distorted"]
boundary              = parsed_args["boundary"]
zmax                 = parsed_args["zmax"]
logadjustment        = parsed_args["logadjustment"]
if distorted == 1
    include("newsets_utils_phi.jl")
elseif distorted == 0
    include("onecapital_utilities.jl")
end

println("=============================================================")

filename = "model.npz" 

filename_ell = "./output/"*dataname*"/Delta_"*string(Delta)*"_distorted_"*string(distorted)*"/logadjustment_"*string(logadjustment)*"_boundary_"*string(boundary)*"/zmax_"*string(zmax)*"_zscale_"*string(zscale)*"/delta_"*string(delta)*"/q_"*string(q)*"_gamma_"*string(gamma)*"_rho_"*string(rho)*"/"
isdir(filename_ell) || mkpath(filename_ell)

#==============================================================================#
#  PARAMETERS
#==============================================================================#
if logadjustment == 0
    logadjustment = false
    println("logadjustment: ", logadjustment)
elseif logadjustment == 1
    logadjustment = true    
    println("logadjustment: ", logadjustment)
end

if boundary == 0
    reflective_boundary = false
    println("reflective_boundary: ", reflective_boundary)
elseif boundary == 1
    reflective_boundary = true
    println("reflective_boundary: ", reflective_boundary)
end

# a11 = 0.07379967217704625
a11 = 0.014*4

if delta == 0.01
    if (rho == 1.0) || (rho == 1.00001)
        alpha = 0.0922
    elseif rho == 1.5
        alpha = 0.108	
    elseif rho == 0.67
        alpha = 0.0819
    elseif rho == 8.0
        alpha = 0.328
    end
elseif delta == 0.015
    if (rho == 1.0) || (rho == 1.00001)
        alpha = 0.1002
    elseif rho == 1.5
        alpha = 0.116
    elseif rho == 0.67
        alpha = 0.08987
    elseif rho == 8.0
        alpha = 0.3359
    end
elseif delta == 0.02
    if (rho == 1.0) || (rho == 1.00001)
        alpha = 0.109
    elseif rho == 1.5
        alpha = 0.125
    elseif rho == 0.67
        alpha = 0.0981
    elseif rho == 8.0
        alpha = 0.347
    end
end
# alpha = 0.05
println(alpha)

# sigma_k = [.00477, .0]*2*sqrt(1.4);
# sigma_z =  [.011, .025]*2*sqrt(1.4);
σ_k = 0.477*2*sqrt(1.4)
sigmaz1 = 0.011*2*sqrt(1.4)
sigmaz2 = 0.025*2*sqrt(1.4)
# σ_k = 0.477*2*1.1
# sigmaz1 = 0.011*2*1.1
# sigmaz2 = 0.025*2*1.1

lsp = [σ_k 0.0; sigmaz1 sigmaz2]
A = lsp * lsp'
c = sqrt(A[2, 2])
b = A[1, 2] / c
a = sqrt(A[1, 1] - b^2)
U = [a b; 0 c]
println("U: ", U)
L = [a 0; b c]
println("L: ", L)
println(U * L - A)

sigma_k = U[1, :] * 0.01 
sigma_z = U[2, :]
println("sigma_k: ", sigma_k)
println("sigma_z: ", sigma_z)

eta = 0.04
beta = 0.04
phi = 8.0

rho1 = 0.0
rho2 = q.^2/dot(sigma_z,sigma_z)/2
# rho2 = q.^2/dot(sigma_z,sigma_z)

# Grid parameters -----------------------------------------------------
II = trunc(Int,200*zscale+1);
zmin = -zmax;

maxit = 1000000;      # maximum number of iterations in the HJB loop
crit  = 10e-6;      # criterion HJB loop

# Initialize model objects -----------------------------------------------------
baseline = Baseline(a11, sigma_z, beta, eta, sigma_k, delta);
technology = Technology(alpha, phi);
robust = Robustness(q, rho1, rho2);
model = OneCapitalEconomy(baseline, technology, robust);
grid = Grid_rz(zmin, zmax, II);
params = FinDiffMethod(maxit, crit, Delta);

# ==============================================================================#
# WITH ROBUSTNESS
# ==============================================================================#

preload = 0

if preload == 1
    preload_kappa = kappa
        preload_rho = rho
        if symmetric_returns == 1
            preload_Delta = 300.0
        else
            if kappa == 0.0
                preload_Delta = 100.0
            else
                preload_Delta = 150.0
            end
        end
        preload_zeta = 0.5
        preload_llim = llim#1.0
        preload_lscale = 3.0#1.0
        preload_zscale = 1.0
        if kappa == 0.0
            preloadname = "/project/lhansen/twocapkpa/output/"*"twocap_re_calib_kappa_0"*"/Delta_"*string(preload_Delta)*"_llim_"*string(preload_llim)*"_lscale_"*string(preload_lscale)*"_zscale_"*string(preload_zscale)*"/kappa_"*string(preload_kappa)*"_zeta_"*string(preload_zeta)*"/gamma_"*string(gamma)*"_rho_"*string(preload_rho)*"/"
        else
            preloadname = "/project/lhansen/twocapkpa/output/"*"twocap_re_calib"*"/Delta_"*string(preload_Delta)*"_llim_"*string(preload_llim)*"_lscale_"*string(preload_lscale)*"_zscale_"*string(preload_zscale)*"/kappa_"*string(preload_kappa)*"_zeta_"*string(preload_zeta)*"/gamma_"*string(gamma)*"_rho_"*string(preload_rho)*"/"
        end
        if symmetric_returns == 1
            preload = npzread(preloadname*"model_sym_HS.npz")
        else
            preload = npzread(preloadname*"model_asym_HS.npz")
        end
        println("preload location : "*preloadname)
        A_x1 = range(-llim,llim,trunc(Int,1000*preload_lscale+1))
        A_x2 = range(-1,1,trunc(Int,200*preload_zscale+1))
        A_rr = range(rmin, stop=rmax, length=II);
        A_zz = range(zmin, stop=zmax, length=JJ);

        println("preload V0 starts")

        itp = interpolate(preload["V"], BSpline(Cubic(Line(OnGrid()))))
        sitp = scale(itp, A_x1, A_x2)
        println("(1,-1): ",sitp(1,-1))
        preloadV0 = ones(II, JJ)
        for i = 1:II
            for j = 1:JJ
                preloadV0[i,j] = sitp(A_rr[i], A_zz[j])
            end
        end
        println("(1,-1): ",preloadV0[end,1])
        println("preload V0 ends")

        println("preload d1 starts")

        itp = interpolate(preload["d1"], BSpline(Cubic(Line(OnGrid()))))
        sitp = scale(itp, A_x1, A_x2)
        println("(1,-1): ",sitp(1,-1))
        preloadd1 = ones(II, JJ)
        for i = 1:II
            for j = 1:JJ
                preloadd1[i,j] = sitp(A_rr[i], A_zz[j])
            end
        end
        println("(1,-1): ",preloadd1[end,1])
        println("preload d1 ends")

        println("preload cons starts")

        itp = interpolate(preload["cons"], BSpline(Cubic(Line(OnGrid()))))
        sitp = scale(itp, A_x1, A_x2)
        println("(1,-1): ",sitp(1,-1))
        preloadcons = ones(II, JJ)
        for i = 1:II
            for j = 1:JJ
                preloadcons[i,j] = sitp(A_rr[i], A_zz[j])
            end
        end
        println("(1,-1): ",preloadcons[end,1])
        println("preload cons ends")
else
    if rho == 1.0
        preloadV0 = -2*ones(grid.I)
        preloadd = 0.03*ones(grid.I)
        preloadcons = 0.03*ones(grid.I)
    else
        preload_logadjustment = Int(logadjustment)
        preload_Delta = 1.0
        preloadname = "./output/"*dataname*"/Delta_"*string(preload_Delta)*"_distorted_"*string(distorted)*"/logadjustment_"*string(preload_logadjustment)*"_boundary_"*string(boundary)*"/zmax_"*string(zmax)*"_zscale_"*string(zscale)*"/delta_"*string(delta)*"/q_"*string(q)*"_gamma_"*string(gamma)*"_rho_"*string(1.0)*"/"
        preload = npzread(preloadname*filename)
        println("preload location : "*preloadname)
        preloadV0 = preload["V"]
        preloadd = preload["d"]
        preloadcons = preload["cons"]
    end
end

println("Compute value function WITH ROBUSTNESS")
times = @elapsed begin
A, V, val, d_F, d_B, hk_F, hz_F, hk_B, hz_B, s1_F, s2_F, lambda_F, s1_B, s2_B, lambda_B, mu_z_distorted_F, mu_z_distorted_B,
    mu_1_F, mu_1_B, mu_z_F, mu_z_B, V0, Vz_B, Vz_F, cF, cB, Vz, zz, dz, uu=
    value_function_onecapital(gamma, rho, model, grid, params, preloadV0, preloadd, preloadcons, foc, clowerlim, logadjustment, reflective_boundary);
println("=============================================================")
end
println("Convegence time (minutes): ", times/60)
g = stationary_distribution(A, grid)

# Define Policies object
policies  = PolicyFunctions(d_F, d_B,
                            -hk_F, -hz_F,
                            -hk_B, -hz_B);

# Construct drift terms under the baseline
mu_1 = (mu_1_F + mu_1_B)/2.;
mu_z = (mu_z_F + mu_z_B)/2.;
hk_dist = (policies.hk_F + policies.hk_B)/2.;
hz_dist = (policies.hz_F + policies.hz_B)/2.;
s1 = (s1_F + s1_B)/2.;
s2 = (s2_F + s2_B)/2.;
d = (policies.d_F + policies.d_B)/2;
hk, hz = -hk_dist, -hz_dist;
mu_z_distorted = (mu_z_distorted_F + mu_z_distorted_B)/2.;

c = alpha*ones(II) - d;

results = Dict("delta" => delta,
"eta" => eta,"a11"=> a11,  "beta" => beta,
"sigma_k" => sigma_k,
"sigma_z" =>  sigma_z, "alpha" => alpha,  "phi" => phi,
"rho1" => rho1, "rho2" => rho2, "q" => q,
"I" => II, "zmax" => zmax, "zmin" => zmin, "zz" => zz, "dz" => dz,
"maxit" => maxit, "crit" => crit, "Delta" => Delta,
"g" => g, "cons" => c,
"V0" => V0, "V" => V, "Vz" => Vz, "Vz_F" => Vz_F, "Vz_B" => Vz_B, "val" => val, "gamma" => gamma, "rho" => rho,
"d" => d,"d_F" => d_F, "d_B" => d_B, "cF" => cF, "cB" => cB,
"hk_F" => hk_F, "hz_F" => hz_F, "hk_B" => hk_B, "hz_B" => hz_B, 
"mu_1_F" => mu_1_F, "mu_1_B" => mu_1_B, "mu_z_F" => mu_z_F, "mu_z_B" => mu_z_B,
"hk" => hk,"hz" => hz, "foc" => foc, "clowerlim" => clowerlim,  "zscale" => zscale, 
"s1_F" => s1_F, "s2_F" => s2_F,  "lambda_F" => lambda_F, "s1_B" => s1_B, "s2_B" => s2_B,  "lambda_B" => lambda_B,
"s1" => s1, "s2" => s2,
"mu_z_distorted" => mu_z_distorted,"mu_z_distorted_F" => mu_z_distorted_F, "mu_z_distorted_B" => mu_z_distorted_B,
"times" => times, "logadjustment" => logadjustment, "reflective_boundary" => reflective_boundary,
"mu_1" => mu_1, "mu_z" => mu_z)

npzwrite(filename_ell*filename, results)
