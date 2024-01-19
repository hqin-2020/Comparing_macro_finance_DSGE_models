#! /bin/bash

Deltaarray=(10.0)

actiontime=1
 
julia_name="onecapital_main.jl"
ela_python_name_dis="onecapital_elasticity_decomposition.py"
ela_python_name_org="onecapital_shock_elasticity.py"
julia_natural_entropy="onecapital_entropy_natural.jl"
julia_reflective_entropy="onecapital_entropy_reflective.jl"

#deltaarray=(0.002 0.02)
deltaarray=(0.01 0.015 0.02)
# deltaarray=(0.002)

#rhoarray=(0.67 0.7  0.8 0.9 1.00001 1.1 1.2 1.3 1.4 1.5)
rhoarray=(1.0)
# rhoarray=(0.67 1.00001 1.5 8.0)

gammaarray=(1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 12.0)
# gammaarray=(4.0)
# zmax=2.0
# zscalearray=(2.0)

zmax=5.0
# zmax=2.0
zscalearray=(2.0)
qarray=(0.1 0.05 0.2 0.3)
distortedarray=(0)
boundaryarray=(0 1)
logadjustmentarray=(1)
# logadjustmentarray=(0)
foc=1
clowerlim=0.0001

for Delta in ${Deltaarray[@]}; do
    for delta in "${deltaarray[@]}"; do
        for rho in "${rhoarray[@]}"; do
            for gamma in "${gammaarray[@]}"; do
                for zscale in "${zscalearray[@]}"; do
                    for q in "${qarray[@]}"; do
                        for distorted in "${distortedarray[@]}"; do
                            for boundary in "${boundaryarray[@]}"; do
                                for logadjustment in "${logadjustmentarray[@]}"; do
                                    count=0

                                    action_name="upper_triangular_four_ambiguous_parameters"

                                    dataname="${action_name}"

                                    mkdir -p ./job-outs/${action_name}/adjustment_${logadjustment}_boundary_${boundary}/Delta_${Delta}_distorted_${distorted}/zmax_${zmax}_zscale_${zscale}/delta_${delta}/rho_${rho}_gamma_${gamma}_q_${q}/

                                    if [ -f ./bash/${action_name}/adjustment_${logadjustment}_boundary_${boundary}/Delta_${Delta}_distorted_${distorted}/zmax_${zmax}_zscale_${zscale}/delta_${delta}/rho_${rho}_gamma_${gamma}_q_${q}/run.sh ]; then
                                        rm ./bash/${action_name}/adjustment_${logadjustment}_boundary_${boundary}/Delta_${Delta}_distorted_${distorted}/zmax_${zmax}_zscale_${zscale}/delta_${delta}/rho_${rho}_gamma_${gamma}_q_${q}/run.sh
                                    fi

                                    mkdir -p ./bash/${action_name}/adjustment_${logadjustment}_boundary_${boundary}/Delta_${Delta}_distorted_${distorted}/zmax_${zmax}_zscale_${zscale}/delta_${delta}/rho_${rho}_gamma_${gamma}_q_${q}/

                                    touch ./bash/${action_name}/adjustment_${logadjustment}_boundary_${boundary}/Delta_${Delta}_distorted_${distorted}/zmax_${zmax}_zscale_${zscale}/delta_${delta}/rho_${rho}_gamma_${gamma}_q_${q}/run.sh

                                    tee -a ./bash/${action_name}/adjustment_${logadjustment}_boundary_${boundary}/Delta_${Delta}_distorted_${distorted}/zmax_${zmax}_zscale_${zscale}/delta_${delta}/rho_${rho}_gamma_${gamma}_q_${q}/run.sh <<EOF
#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --job-name=D_${Delta}_distorted_${distorted}_boundary_${boundary}_delta_${delta}_gamma_${gamma}_q_${q}_rho_${rho}_${action_name}
#SBATCH --output=./job-outs/${action_name}/adjustment_${logadjustment}_boundary_${boundary}/Delta_${Delta}_distorted_${distorted}/zmax_${zmax}_zscale_${zscale}/delta_${delta}/rho_${rho}_gamma_${gamma}_q_${q}/run.out
#SBATCH --error=./job-outs/${action_name}/adjustment_${logadjustment}_boundary_${boundary}/Delta_${Delta}_distorted_${distorted}/zmax_${zmax}_zscale_${zscale}/delta_${delta}/rho_${rho}_gamma_${gamma}_q_${q}/run.err
#SBATCH --time=0-23:00:00
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

module load julia/1.7.3
module load python/anaconda-2020.11

echo "\$SLURM_JOB_NAME"

echo "Program starts \$(date)"
start_time=\$(date +%s)

srun julia /project/lhansen/Comparing_macro_finance_DSGE_models/onecap_model_with_structure_ambiguity/src/$julia_name  --Delta ${Delta} --delta ${delta} --gamma ${gamma} --rho ${rho} --q ${q} --dataname ${dataname} --zscale ${zscale} --foc ${foc} --clowerlim ${clowerlim}  --distorted ${distorted} --zmax ${zmax} --boundary ${boundary} --logadjustment ${logadjustment}
python3 /project/lhansen/Comparing_macro_finance_DSGE_models/onecap_model_with_structure_ambiguity/src/$ela_python_name_dis  --Delta ${Delta} --delta ${delta} --gamma ${gamma} --rho ${rho} --q ${q} --dataname ${dataname} --zscale ${zscale} --distorted ${distorted} --zmax ${zmax} --boundary ${boundary} --logadjustment ${logadjustment}
python3 /project/lhansen/Comparing_macro_finance_DSGE_models/onecap_model_with_structure_ambiguity/src/$ela_python_name_org  --Delta ${Delta} --delta ${delta} --gamma ${gamma} --rho ${rho} --q ${q} --dataname ${dataname} --zscale ${zscale} --distorted ${distorted} --zmax ${zmax} --boundary ${boundary} --logadjustment ${logadjustment}
srun julia /project/lhansen/Comparing_macro_finance_DSGE_models/onecap_model_with_structure_ambiguity/src/$julia_natural_entropy  --Delta ${Delta} --delta ${delta} --gamma ${gamma} --rho ${rho} --q ${q} --dataname ${dataname} --zscale ${zscale} --foc ${foc} --clowerlim ${clowerlim}  --distorted ${distorted} --zmax ${zmax} --boundary ${boundary} --logadjustment ${logadjustment}
srun julia /project/lhansen/Comparing_macro_finance_DSGE_models/onecap_model_with_structure_ambiguity/src/$julia_reflective_entropy  --Delta ${Delta} --delta ${delta} --gamma ${gamma} --rho ${rho} --q ${q} --dataname ${dataname} --zscale ${zscale} --foc ${foc} --clowerlim ${clowerlim}  --distorted ${distorted} --zmax ${zmax} --boundary ${boundary} --logadjustment ${logadjustment}

echo "Program ends \$(date)"
end_time=\$(date +%s)
elapsed=\$((end_time - start_time))

eval "echo Elapsed time: \$(date -ud "@\$elapsed" +'\$((%s/3600/24)) days %H hr %M min %S sec')"

EOF
                                    count=$(($count + 1))
                                    sbatch ./bash/${action_name}/adjustment_${logadjustment}_boundary_${boundary}/Delta_${Delta}_distorted_${distorted}/zmax_${zmax}_zscale_${zscale}/delta_${delta}/rho_${rho}_gamma_${gamma}_q_${q}/run.sh
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done