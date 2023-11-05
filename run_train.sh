datasets='adult'
synthesizers='margctgan'
subsets='40 160 640 1280 -1'
model_random_states='1000 2000 3000'
dataset_random_states='1000 2000 3000'
nsynth=10
synth_size='-1 20000'
device='cuda'
ep=300

# DRS - Dataset Random Seed
# MRS - Model Random Seed


exp_name=$(date +'%d.%m.%Y')

for data in ${datasets}; do
for drs in ${dataset_random_states}; do
 for subset in ${subsets}; do
    for model in ${synthesizers}; do
        for mrs in ${model_random_states}; do
            if [[ "$model" == "margctgan" ]];
                then
                    variant='random_orthogonal_matrix'
                    condition_vector='True'
                    stats='mean_and_stddev'
                    n_components='-1'
                    unique_name=c_${condition_vector}+v_${variant}+n_${n_components}+s_${stats}
                    name=${exp_name}2/${data}/DRS${drs}/subset${subset}/${model}-${unique_name}/MRS${mrs}/epoch${ep}
                    echo "Running: ${name}"
                    python synthesizers/${model}/${model}.py \
                        -name ${name}\
                        -data ${data} \
                        -ep ${ep} \
                        -s ${mrs} \
                        --dataset_random_state ${drs} \
                        --dataset_dir ./data \
                        --report_dir ./reports \
                        --train 'false' \
                        --sample 'true' \
                        --evaluate 'true' \
                        --subset_size ${subset} \
                        --synth_size ${synth_size} \
                        --nsynth ${nsynth} \
                        --device ${device} \
                        --variant ${variant} \
                        --n_components ${n_components} \
                        --stats ${stats} \
                        --condition_vector ${condition_vector}
            else
                name=${exp_name}/${data}/DRS${drs}/subset${subset}/${model}/MRS${mrs}/epoch${ep}
                echo "Running: ${name}"
                python synthesizers/${model}/${model}.py \
                    -name ${name}\
                    -data ${data} \
                    -ep ${ep} \
                    -s ${mrs} \
                    --dataset_random_state ${drs} \
                    --dataset_dir ./data \
                    --train 'true' \
                    --sample 'true' \
                    --subset_size ${subset} \
                    --synth_size ${synth_size} \
                    --nsynth ${nsynth} \
                    --device ${device}
            fi
        done
    done
done
done
done

                