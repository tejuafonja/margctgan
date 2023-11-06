random_state="1000 2000 3000"
# random_state="3000"
subsets="40 80 160 320 640 1280 2560 5120 10240 20480 -1"
# subsets="160"
metrics="all_models_test"
# metrics="efficacy_test histogram_intersection closeness_approximation associations_difference"
synthesizers="wgan_cond_marg-feat__fixed_random_matrix-stat__stddev-dim__na"
# synthesizers="sdv_ctgan sdv_tvae tablegan"
# synthesizers="sdv_ctgan"
# datasets="adult texas news census"
datasets="adult"

# data/fake_samples/RS1000/subset-1/02.10.2023/epoch300/adult/random_projection-na-stddev-weighted-1.0-ctgan


eval_retries=5
ep=300
# ep=1000
exp_name=02.10.2023
sample_size=20000

for metric_name in ${metrics}; do
for rs in ${random_state}; do
    for data in ${datasets}; do
        for synth in ${synthesizers}; do
            for subset in ${subsets}; do
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
                    name=RS${rs}/subset${subset}/${exp_name}/epoch${ep}
                    python metrics/evaluate.py \
                        -name ${name}\
                        -data ${data} \
                        -s ${rs} \
                        --dataset_dir ./data \
                        --synthesizer ${synth} \
                        --subset_size ${subset} \
                        --eval_retries ${eval_retries} \
                        --metric_name ${metric_name} \
                        --sample_size ${sample_size} \
                        --metric_group ${metric_group} \
                        --overwrite_results "True"
                    sleep 5
                done
            done
        done
    done
done