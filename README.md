# MargCTGAN
[![LICENSE](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7.0-orange)](https://pytorch.org/)

![image](averaged_metrics_result.png)
*The **X-axis** represents the size of the training dataset, with "all" indicating the full dataset size. **Real data (reference)** corresponds to the metrics directly measured on the real (train vs. test) data, serving as the reference (oracle score) for optimal performance.*


This repository contains the implementation for "MargCTGAN: A ``Marginally'' Better CTGAN for the Low Sample Regime" Published at [GCPR 2023](https://link.springer.com/chapter/10.1007/978-3-031-54605-1_34). This paper was also presented at [ICML 2023 Deploying Generative AI Workshop](https://openreview.net/pdf?id=4apndCCMv4).

*Authors: Tejumade Afonja, Dingfan Chen, and Mario Fritz*

Contact: Tejumade Afonja ([tejumade.afonja@cispa.de](mailto:tejumade.afonja@cispa.de))


## Requirements
This implementation is based on [PyTorch](https://www.anaconda.com/download/) (tested for version 1.7.0). Please refer to [requirements.txt](requirements.txt) for the other required packages and version.  

## Datasets
The implementation supports the following datasets:
- [Adult](https://archive.ics.uci.edu/ml/datasets/adult), [Census](https://archive.ics.uci.edu/dataset/117/census+income+kdd), [News](https://archive.ics.uci.edu/ml/datasets/online+news+popularity), and [Texas](https://github.com/spring-epfl/synthetic_data_release/blob/master/data/texas.csv).

You can download the data by running the notebooks in `data/<dataset_name>/<dataset_name>-download.ipynb`.

## Running Experiments
### API (Run experiments using the default configurations).
Change to the `synthesizers/<name_of_model>` directory and run the code snippet below (after updating the placeholders).
```code 
python <name_of_model_main_script>.py \
    -name <name_of_experiment> \
    -data <dataset_name> \
    -ep <number_of_epoch> \
    -s <seed> \
    --train "True" \
    --evaluate "True" \
    --sample "True" \
    --subset_size <size_of_real_dataset> \
    --sample_size <size_of_synthetic_dataset> \
    --eval_retries <how_many_times_to_rerun_evaluation>
```

## Citation
```bibtex
@inproceedings{afonja2023margctgan,
  title={MargCTGAN: A “Marginally” Better CTGAN for the Low Sample Regime},
  author={Afonja, Tejumade and Chen, Dingfan and Fritz, Mario},
  booktitle={DAGM German Conference on Pattern Recognition},
  pages={524--537},
  year={2023},
  organization={Springer}
}
```

## Acknowledgements
Our implementation uses the source code from the following repositories:
- [Synthetic Data Vault](https://github.com/sdv-dev)
- [Table Evaluator](https://github.com/Baukebrenninkmeijer/table-evaluator)
