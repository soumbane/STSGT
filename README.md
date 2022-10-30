# Spatial–Temporal Synchronous Graph Transformer network (STSGT) for COVID-19 forecasting

This is the official implementation of the paper "[Spatial–Temporal Synchronous Graph Transformer network (STSGT) for COVID-19 forecasting](https://www.sciencedirect.com/science/article/pii/S2352648322000824)" that was presented at IEEE/ACM CHASE 2022 conference and is published in Elsevier Smart Health Journal (2022).

## Requirements
* Python >= 3.6
* PyTorch >= 1.8.0
* See requirements.txt (All of them are not required)

## Get Started
The following steps are required to replicate our work:

1. Download datasets.
* JHU Dataset - Download [JHU COVID time-series data](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series) (download `time_series_covid19_confirmed_US.csv` for daily US infected cases and `time_series_covid19_deaths_US.csv` for daily US death cases) and save in `data/COVID_JHU` directory. This project used `Mar 15,2020 - Nov 30,2021` for analysis. 
* NYT Dataset - Download [NYT COVID time-series data](https://github.com/nytimes/covid-19-data) (download `us-states.csv` for daily US infected and death cases) and save in `data/COVID_NYT` directory. This project used `Mar 18,2020 - Nov 30,2021` for analysis. 

2. Generate Feature Matrix (X) and Adjacency Matrix (W) from downloaded datasets.
* JHU Dataset (US) - Inside the folder `data/COVID_JHU`, run the file `Generate_51_states_X_W.ipynb` to generate X and W matrix for 50 states of US and Washington D.C. (51 nodes of graph).
* JHU Dataset (Michigan) - Inside the folder `data/COVID_JHU`, run the file `Generate_51_states_X_W_Michigan.ipynb` to generate X and W matrix for 83 counties of the state of Michigan (83 nodes of graph).
* NYT Dataset (US) - Inside the folder `data/COVID_NYT`, run the file `Generate_51_states_X_W_NYT.ipynb` to generate X matrix for 50 states of US and Washington D.C. (51 nodes of graph). We used the same adjacency matrix (W) as generated using JHU dataset.

3. Generate Train, Validation and Test datasets from the generated X matrix.
* We divided the entire dataset in chronological order with 80% training, 10% validation and 10% testing.
* Run the file `generate_training_data.py` to generate the processed files `train.npz, val.npz, test.npz` from X matrix and save the processed files in `data/COVID_JHU/processed` or `data/COVID_NYT/processed`. Use the `confirmed` or `deaths` in the argument to generate infected and death cases processed files.
```
python generate_training_data.py --traffic_df_filename "data/COVID_JHU/covid19_deaths_US_51_states_X_matrix_final.csv" 
```

## Training

1. Define paths and hyper-parameters in configuration files.
* Refer to the files `COVID_JHU.conf` and `COVID_NYT.conf` for the data paths and hyper-parameters used for training and testing. 
* The `sensors_distance` in the config files indicate the path to the adjacency matrix W.

2. Train the model



## Cite
Please cite our paper if you find this work useful for your research:
```
@article{BANERJEE2022100348,
title = {Spatial–Temporal Synchronous Graph Transformer network (STSGT) for COVID-19 forecasting},
journal = {Smart Health},
volume = {26},
pages = {100348},
year = {2022},
issn = {2352-6483},
doi = {https://doi.org/10.1016/j.smhl.2022.100348},
url = {https://www.sciencedirect.com/science/article/pii/S2352648322000824},
author = {Soumyanil Banerjee and Ming Dong and Weisong Shi}
}
```

