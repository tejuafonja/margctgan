dataset_random_states="1000"
model_random_states="1000"
# random_state="3000"
subsets="640"
# subsets="160"
metrics="associations_difference"
# metrics="efficacy_test histogram_intersection closeness_approximation associations_difference"
synthesizers="gmm"
# synthesizers="sdv_ctgan sdv_tvae tablegan"
# synthesizers="sdv_ctgan"
# datasets="adult texas news census"
datasets="adult"

eval_retries=10
ep=300
# ep=1000
exp_name=06.11.2023
synth_size=20000

for metric_name in ${metrics}; do
    for data in ${datasets}; do
        for drs in ${dataset_random_states}; do
            for subset in ${subsets}; do
                for model in ${synthesizers}; do
                    for mrs in ${model_random_states}; do
                        if [ ${metric_name} = "efficacy_test" ];
                        then
                            metric_group="utility"
                        elif [ ${metric_name} = "all_models_test" ];
                        then
                            metric_group="utility"
                        elif [ ${metric_name} = "histogram_intersection" ];
                        then
                            metric_group="marginal"
                        elif [ ${metric_name} = "jensonshannon_distance" ];
                        then
                            metric_group="marginal"
                        elif [ ${metric_name} = "wasserstein_distance" ];
                        then
                            metric_group="marginal"
                        elif [ ${metric_name} = "column_correlation" ];
                        then
                            metric_group="marginal"
                        elif [ ${metric_name} = "total_variation_distance" ];
                        then
                            metric_group="marginal"
                        elif [ ${metric_name} = "jaccard_similarity" ];
                        then
                            metric_group="marginal"
                        elif [ ${metric_name} = "chebychev_chi2" ];
                        then
                            metric_group="marginal"
                        elif [ ${metric_name} = "closeness_approximation" ];
                        then
                            metric_group="joint"
                        elif [ ${metric_name} = "likelihood_approximation" ];
                        then
                            metric_group="joint"
                        elif [ ${metric_name} = "associations_difference" ];
                        then
                            metric_group="column_pair"
                        fi
                        name=${exp_name}/${data}/DRS${drs}/subset${subset}/${model}/MRS${mrs}/epoch${ep}
                        python metrics/evaluate.py \
                            -name ${name}\
                            -data ${data} \
                            -s ${drs} \
                            --dataset_dir ./data \
                            --synthesizer ${model} \
                            --subset_size ${subset} \
                            --eval_retries ${eval_retries} \
                            --metric_name ${metric_name} \
                            --synth_size ${synth_size} \
                            --metric_group ${metric_group} \
                            --overwrite_results "True"
                        sleep 2
                    done
                done
            done
        done
    done
done