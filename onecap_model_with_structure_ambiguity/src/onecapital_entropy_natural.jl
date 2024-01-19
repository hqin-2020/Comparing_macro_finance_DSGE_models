using NPZ
using LinearAlgebra
using SparseArrays

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
rhoname = "relative_entropy_natural.npz"
filename_ell = "./output/"*dataname*"/Delta_"*string(Delta)*"_distorted_"*string(distorted)*"/logadjustment_"*string(logadjustment)*"_boundary_"*string(boundary)*"/zmax_"*string(zmax)*"_zscale_"*string(zscale)*"/delta_"*string(delta)*"/q_"*string(q)*"_gamma_"*string(gamma)*"_rho_"*string(rho)*"/"
isdir(filename_ell) || mkpath(filename_ell)
println(filename_ell)
npz = npzread(filename_ell*filename)

zmax = npz["zmax"];
zmin = npz["zmin"];
II = trunc(Int,200*zscale+1)
dz = (zmax - zmin)/(II-1);
dz2 = dz*dz;
t2 = dot(npz["sigma_z"], npz["sigma_z"])/(2*dz2);

println("=============================================================")

A = spdiagm( 0 => ones(II),
             1 => ones(II-1),
             -1 => ones(II-1),
             2 => ones(II-2),
             -2 => ones(II-2));
Aval = zeros(nnz(A))

mu_z_F = npz["mu_z_F"] + npz["mu_z_distorted_F"];
mu_z_B = npz["mu_z_B"] + npz["mu_z_distorted_B"];

# Adding reflection boundary in I dimension
c_ = max.(mu_z_F, 0.)/dz .+ t2;
d_ = - max.(mu_z_F, 0.)/dz + min.(mu_z_B, 0.)/dz .- 2*t2;
e_ = -min.(mu_z_B, 0.)/dz .+ t2;
c_2 = zeros(II);
e_2 = zeros(II);

c_[end] = 0.0;
d_[end] = (mu_z_B./dz .+ t2)[end];
e_[end] = (-mu_z_B./dz .- 2*t2)[end];
e_2[end] = t2;

c_[1] = (mu_z_F./dz .- 2*t2)[1];
d_[1] = (-mu_z_F./dz .+ t2)[1];
e_[1] = 0.0;
c_2[1] = t2;

create_Aval_natural!(Aval, d_, c_, e_, c_2, e_2, II)
A.nzval .= Aval;

middle_idx = Int((II+1)/2)
A[:,middle_idx].=-1
println("solve for structured linear system ")
strucuted_sol = A \ (-(npz["s1"].^2 + npz["s2"].^2)/2)[:]
structured_relative_entropy = sqrt(strucuted_sol[middle_idx]*2)
println("structured_relative_entropy = ", structured_relative_entropy)
strucuted_sol[middle_idx]=0
structured_test_function = reshape(strucuted_sol, II, 1)

println("=============================================================")
A = spdiagm( 0 => ones(II),
             1 => ones(II-1),
             -1 => ones(II-1),
             2 => ones(II-2),
             -2 => ones(II-2));
      Aval = zeros(nnz(A))

mu_z_F = npz["mu_z_F"] + npz["mu_z_distorted_F"] + (1-gamma) *(npz["hk_F"]*npz["sigma_z"][1] + npz["hz_F"]*npz["sigma_z"][2]);
mu_z_B = npz["mu_z_B"] + npz["mu_z_distorted_B"] + (1-gamma) *(npz["hk_B"]*npz["sigma_z"][1] + npz["hz_B"]*npz["sigma_z"][2]);

#CONSTRUCT MATRIX A
c_ = max.(mu_z_F, 0.)/dz .+ t2;
d_ = - max.(mu_z_F, 0.)/dz + min.(mu_z_B, 0.)/dz .- 2*t2;
e_ = -min.(mu_z_B, 0.)/dz .+ t2;
c_2 = zeros(II);
e_2 = zeros(II);

c_[end] = 0.0;
d_[end] = (mu_z_B./dz .+ t2)[end];
e_[end] = (-mu_z_B./dz .- 2*t2)[end];
e_2[end] = t2;

c_[1] = (mu_z_F./dz .- 2*t2)[1];
d_[1] = (-mu_z_F./dz .+ t2)[1];
e_[1] = 0.0;
c_2[1] = t2;

create_Aval_natural!(Aval, d_, c_, e_, c_2, e_2, II)
A.nzval .= Aval;

middle_idx = Int((II+1)/2)
A[:,middle_idx].=-1
println("solve for unstructured linear system ")
unstructured_sol = A \ -(( ((1-gamma)*npz["hk"]).^2 + ((1-gamma)*npz["hz"]).^2) /2)[:]
unstructured_relative_entropy = sqrt(unstructured_sol[middle_idx]*2)
println("q_update = ", unstructured_relative_entropy)
unstructured_sol[middle_idx]=0
unstructured_test_function = reshape(unstructured_sol, II, 1)

println("=============================================================")

results = Dict("structured_relative_entropy" => structured_relative_entropy, "unstructured_relative_entropy" => unstructured_relative_entropy, "structured_test_function" => structured_test_function, "unstructured_test_function" => unstructured_test_function)
npzwrite(filename_ell*rhoname, results)