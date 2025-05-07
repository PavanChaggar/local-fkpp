# Code for "Personalised Regional Modelling Predicts Tau Progression in the Human Brain", Chaggar et al 2024

This repository contains the code used in the paper Chaggar et al., 2024 
for the analysis of ADNI data. 

## Code installation 

We use the [Julia Progamming Language](https://github.com/JuliaLang/julia) 
for this project. The full package environment is uploaded in the `Project.toml` 
and `Manifest.toml` and can be used to instantiate the dependencies used by 
this project.

## Data analysis

ADNI data were downloaded from the LONI portal 
[https://adni.loni.usc.edu/](https://adni.loni.usc.edu/). Amyloid and 
tau PET data were last accessed 18/12/23. We classify subjects as into three 
cohorts depending on amyloid and tau status. Amyloid status is determined provided 
by ADNI and the script for determining individual amyloid status is 
[`adni/data/data-processing.jl`](https://github.com/PavanChaggar/local-fkpp/tree/main/adni/data/data-processing.jl). The script for calculating the demographics
of each cohort used in the paper is in 
[`adni/data/demographics.jl`](https://github.com/PavanChaggar/local-fkpp/tree/main/adni/data/demographics.jl). 

## Inference
 
We perform inference over four models, the local FKPP, global FKPP, logistic 
and diffusion models.

The code used for loading the connectome and ADNI data 
is in [`adni/inference/inference-preamble.jl`](https://github.com/PavanChaggar/local-fkpp/blob/main/adni/inference/inference-preamble.jl) and is imported 
into the separate analysis scripts for each model.

For the local FKPP model, we run inference for three cohorts, 
A+T+, A+T- and A-T-, for the remaining models, we only run inference on the A+T+ groups. 
For each of these models, inference scripts can be found in 
`adni/inference` directory. For example, inference for 
the A+T+ group using the local FKPP model is found in 
[`adni/inference/local-fkpp/taupos.jl`](https://github.com/PavanChaggar/local-fkpp/blob/main/adni/inference/local-fkpp/taupos.jl). 

Additionally, for all models, we run inference using a test/train data split, using only the first three scans for training. The scripts for these are found in the model inference directories and suffixed with `-three`. 

## Visualisation

The visualisation the posterior analysis can be found in `visualisation`. 

The panels in Figure 1 of the manuscript are created using 
the scripts in `visualisation/models`. 

The panels in Figure 2 of the manuscript are created using [`visualisation/inference/model-selection/model-selection-pos.jl`](https://github.com/PavanChaggar/local-fkpp/blob/main/visualisation/inference/model-selection/model-selection-pos.jl). 

The panels in Figure 3 of the manuscript for the ADNI posterior distributions
are created using [`visualisation/posteriors/posteriors.jl](https://github.com/PavanChaggar/local-fkpp/blob/main/visualisation/inference/posteriors/posteriors.jl). 

The panels in Figure 4 of the manscripts are generated using ['visualisation/inference/model-selection/out-sample-three-pos.jl](https://github.com/PavanChaggar/local-fkpp/blob/main/visualisation/inference/model-selection/out-sample-three-pos.jl). 

## Revisions 

Code used for analysis stemming from the peer-review process is in the [`revisions`](https://github.com/PavanChaggar/local-fkpp/tree/revisions) branch. The posterior distributions stemming from this analysis are in the directors `adni/chains-revisions/local-fkpp` for `pvc-ic` and `wm`, corresponding to analysis on PVC data with inferior cerebllar reference reigion and non-PVC data with eroded white matter reference region, respecitvely.

Code for the reference region analysis is documented in the notebook [`reference_region_analysis`](https://github.com/PavanChaggar/local-fkpp/blob/revisions/adni/inference/analysis/notebooks/reference_region_analysis.ipynb).

## Citation

Chaggar, Pavanjit, Jacob W. Vogel, Alexa Pichet-Binette, Travis B. Thompson, Olof B. Strandberg, Erik Stomrud, Niklas Mattsson-Carlgren et al. "Personalised Regional Modelling Predicts Tau Progression in the Human Brain." bioRxiv (2023): 2023-09.

```
@article{chaggar2023personalised,
  title={Personalised Regional Modelling Predicts Tau Progression in the Human Brain},
  author={Chaggar, Pavanjit and Vogel, Jacob W and Pichet-Binette, Alexa and Thompson, Travis B and Strandberg, Olof B and Stomrud, Erik and Mattsson-Carlgren, Niklas and Karlsson, Linda and Jbabdi, Saad and Magon, Stefano and others},
  journal={bioRxiv},
  pages={2023--09},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```