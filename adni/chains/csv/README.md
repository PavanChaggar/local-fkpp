# ADNI Posterior Chains 

Files contained here contain the population and individual level chains derived from ADNI data. Present here are chains for the A+T+, A+T-, A-T- biomarker groups using the standard inferior cerebellum reference region, white matter reference region `-wm`, PVC with inferior cerebellar reference `-pvc-ic` and shuffled analysis with inferior cerebellum reference `-shuffled`. The number of chains (N) and samples (S) is denoted `-NxS`.

Column names for parameters are provided in the table below. 

| Parameter name    | Model parameter |
| --------------    | --------------- |
|       σ           | Observation noise |
| Pm                |  Population level mean transport rate   |
| Ps                |  Population level std transport rate    |
| Am                |  Population level mean production rate  |
| As                |  Population level std production rate   |
| ρ[i]              |  Transport rate for subject i   |
| α[i]              |  Production rate for subject i  |


The files `adni-pst-taupos-4x2000`, `adni-pst-tauneg-4x2000` and `adni-pst-abneg-4x2000.csv` correspond to the ADNI posterior distributions in Figure 4a of the manuscript and Figure S5, for the A+T+, A+T-, A-T- groups, respecitively. The files `adni-pst-taupos-shuffled-10x1000` 
and `adni-pst-tauneg-shuffled-10x1000` correspond to to the shuffled posterior distributions in Figure 4c and Figure 4d of the manuscript, repsectively.

The files `adni-pst-taupos-pvc-ic-1x2000`, `adni-pst-tauneg-pvc-ic-1x2000` and `adni-pst-abneg-pvc-ic-1x2000.csv` correspond to the ADNI posterior distributions in Figure S8a of the manuscript. Files `adni-pst-taupos-wm-1x2000`, `adni-pst-tauneg-wm-1x2000` and `adni-pst-abneg-wm-1x2000.csv` correspond to the posterior distributions in Figure S8b of the manuscript. 

For all visualisations, the box plot show the median and 1.5 $\times$ interquartile range. 
The BioFINDER-2 data for Figure 4b, Figure S6 and Figure S9 can be found [here](https://github.com/PavanChaggar/local-fkpp/tree/main/biofinder/chains/csv).
