# BioFINDER-2 Posterior Chains 

Files contained here contain the population and individual level chains derived from BioFINDER-2 data. Present here are chains for the A+T+, A+T-, A-T- biomarker groups using the standard inferior cerebellum reference region on the DK atlas and on the Schaefer-200 atlas `-schaefer`. The number of chains (N) and samples (S) is denoted `-NxS`. 

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

The files `bf-pst-taupos-4x1000`, `bf-pst-tauneg-4x1000` and `bf-pst-abneg-4x1000`correspond to the BF posterior distributions in Figure 4b and Figure S6, for the A+T+, A+T-, A-T- groups, respecitively.

The files `bf-pst-taupos-schaefer-4x1000`, `bf-pst-tauneg-schaefer-4x1000` and `bf-pst-abneg-schaefer-4x1000`correspond to the BF posterior distributions in Figure S9 with the Schaefer-200 atlas, for the A+T+, A+T-, A-T- groups, respecitively.

The BioFINDER-2 data for Figure 4b, Figure S6 and Figure S9 can be found [here](https://github.com/PavanChaggar/local-fkpp/tree/main/biofinder/chains/csv).