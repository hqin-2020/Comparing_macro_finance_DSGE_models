#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --job-name=D_10.0_distorted_0_boundary_1_delta_0.02_gamma_9.0_q_0.05_rho_1.0_lower_triangular_two_ambiguous_parameters
#SBATCH --output=./job-outs/lower_triangular_two_ambiguous_parameters/adjustment_1_boundary_1/Delta_10.0_distorted_0/zmax_5.0_zscale_2.0/delta_0.02/rho_1.0_gamma_9.0_q_0.05/run.out
#SBATCH --error=./job-outs/lower_triangular_two_ambiguous_parameters/adjustment_1_boundary_1/Delta_10.0_distorted_0/zmax_5.0_zscale_2.0/delta_0.02/rho_1.0_gamma_9.0_q_0.05/run.err
#SBATCH --time=0-23:00:00
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

module load julia/1.7.3
module load python/anaconda-2020.11

echo "$SLURM_JOB_NAME"

echo "Program starts $(date)"
start_time=$(date +%s)

srun julia /project/lhansen/Comparing_macro_finance_DSGE_models/onecap_model_with_structure_ambiguity/src/onecapital_main.jl  --Delta 10.0 --delta 0.02 --gamma 9.0 --rho 1.0 --q 0.05 --dataname lower_triangular_two_ambiguous_parameters --zscale 2.0 --foc 1 --clowerlim 0.0001  --distorted 0 --zmax 5.0 --boundary 1 --logadjustment 1
python3 /project/lhansen/Comparing_macro_finance_DSGE_models/onecap_model_with_structure_ambiguity/src/onecapital_elasticity_decomposition.py  --Delta 10.0 --delta 0.02 --gamma 9.0 --rho 1.0 --q 0.05 --dataname lower_triangular_two_ambiguous_parameters --zscale 2.0 --distorted 0 --zmax 5.0 --boundary 1 --logadjustment 1
python3 /project/lhansen/Comparing_macro_finance_DSGE_models/onecap_model_with_structure_ambiguity/src/onecapital_shock_elasticity.py  --Delta 10.0 --delta 0.02 --gamma 9.0 --rho 1.0 --q 0.05 --dataname lower_triangular_two_ambiguous_parameters --zscale 2.0 --distorted 0 --zmax 5.0 --boundary 1 --logadjustment 1
srun julia /project/lhansen/Comparing_macro_finance_DSGE_models/onecap_model_with_structure_ambiguity/src/onecapital_entropy_natural.jl  --Delta 10.0 --delta 0.02 --gamma 9.0 --rho 1.0 --q 0.05 --dataname lower_triangular_two_ambiguous_parameters --zscale 2.0 --foc 1 --clowerlim 0.0001  --distorted 0 --zmax 5.0 --boundary 1 --logadjustment 1
srun julia /project/lhansen/Comparing_macro_finance_DSGE_models/onecap_model_with_structure_ambiguity/src/onecapital_entropy_reflective.jl  --Delta 10.0 --delta 0.02 --gamma 9.0 --rho 1.0 --q 0.05 --dataname lower_triangular_two_ambiguous_parameters --zscale 2.0 --foc 1 --clowerlim 0.0001  --distorted 0 --zmax 5.0 --boundary 1 --logadjustment 1

echo "Program ends $(date)"
end_time=$(date +%s)
elapsed=$((end_time - start_time))

eval "echo Elapsed time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"

