# MargCTGAN
[![LICENSE](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7.0-orange)](https://pytorch.org/)

![image](averaged_metrics_result.png)
*The **X-axis** represents the size of the training dataset, with "all" indicating the full dataset size. **Real data (reference)** corresponds to the metrics directly measured on the real (train vs. test) data, serving as the reference (oracle score) for optimal performance.*


This repository contains the implementation for ["MargCTGAN: A ``Marginally'' Better CTGAN for the Low Sample Regime" (ICML 2023 Deploying Generative AI Workshop)](https://openreview.net/pdf?id=4apndCCMv4).

*Authors: Tejumade Afonja, Dingfan Chen, and Mario Fritz*

Contact: Tejumade Afonja ([tejumade.afonja@cispa.de](mailto:tejumade.afonja@cispa.de))


## Requirements
This implementation is based on [PyTorch](https://www.anaconda.com/download/) (tested for version 1.7.0). Please refer to [requirements.txt](requirements.txt) for the other required packages and version.  

## Datasets
The implementation supports the following datasets:
- [Adult](https://archive.ics.uci.edu/ml/datasets/adult), [Census](https://archive.ics.uci.edu/dataset/117/census+income+kdd), [News](https://archive.ics.uci.edu/ml/datasets/online+news+popularity), and [Texas](https://github.com/spring-epfl/synthetic_data_release/blob/master/data/texas.csv).

You can download the data by running the notebooks in `data/<dataset_name>/<dataset_name>-download.ipynb`.

## Running Experiments
Run the following in the root directory.

`export PYTHONPATH=$PWD `

### API (Run experiments using the default configurations).
Change to the `synthesizers/<name_of_model>` directory and run the code snippet below (after updating the placeholders).
```code 
python <name_of_model_main_script>.py \
    -name <name_of_experiment> \
    -data <name_of_dataset> \
    -ep <number_of_epoch> \
    -s <seed> \
    --train "True" \
    --sample "True" \
    --subset_size <size_of_real_dataset> \
    --sample_size <size_of_synthetic_dataset> \
    --eval_retries <how_many_times_to_rerun_evaluation>
```
This will create synthetic dataset for the model under `data/fake_sample/<name_of_dataset>/`. The results of the experiments can be found under `synthesizers/results/<name_of_experiment>`.
### Evaluation
From the root directory, change to the `metrics/` directory and run the code snippet below (after updating the placeholders).

```code
python evaluate.py \
    -name <name_of_experiment> \
    -data <name_of_dataset> \
    -s <seed> \
    --synthesizer <name_of_model> \
    --subset_size <size_of_real_dataset> \
    --sample_size <size_of_synthetic_dataset_generated_by_trained_model> \
    --metric_name <name_of_metric> \
    --metric_group <group_of_metric> \
    --eval_retries <number_of_synthetic_sample_dataset_generated> \
    --overwrite_results <whether_or_not_to_overwrite_results>
```
Running the script above will create a new results folder in the metrics folder `metrics/results/<name_of_dataset>/<name_of_model>\<name_of_experiment>/`. The `summary_*` in the metrics results folder corresponds to the averaged result.
In the same folder, you will find different files corresponding to: 
 - Real data evaluation (real `traintest`). Used for comparison
 - Fake data evaluation (`faketrain` and `faketest`). 
 Usually you are more interested in the `faketest` csv files. For example, for the utility-based metrics, this corresponds to training on the fake data and evaluating on the real test data.

## Citation
```bibtex
@article{afonja2023margctgan,
  title={MargCTGAN: A" Marginally''Better CTGAN for the Low Sample Regime},
  author={Afonja, Tejumade and Chen, Dingfan and Fritz, Mario},
  journal={arXiv preprint arXiv:2307.07997},
  year={2023}
}
```

## Acknowledgements
Our implementation uses the source code from the following repositories:
- [Synthetic Data Vault](https://github.com/sdv-dev)
- [Table Evaluator](https://github.com/Baukebrenninkmeijer/table-evaluator)
