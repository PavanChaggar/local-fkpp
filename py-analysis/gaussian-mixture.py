import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from scipy.stats import norm 
import pandas as pd

all_data = np.genfromtxt('data-pvc-wm.txt', delimiter='')
df =  pd.DataFrame(columns = ["region","C0_mean","C1_mean","C0_cov","C1_cov","cutoff"])


for i in np.arange(0, 72):
    data = all_data[i,:]

    model1 = GaussianMixture(n_components=1,random_state=123)
    model1.fit(data.reshape(-1,1))

    model2 = GaussianMixture(n_components=2,random_state=123)
    model2.fit(data.reshape(-1,1))

    aic1 = model1.aic(data.reshape(-1,1))
    aic2 = model2.aic(data.reshape(-1,1)) 
    if aic1 > aic2:
        mean_1, mean_2 =  model2.means_[0][0], model2.means_[1][0]
        var_1, var_2 = model2.covariances_[0][0][0], model2.covariances_[1][0][0]
        if mean_1 < mean_2:
            print([mean_1, mean_2])
            vals = np.arange(mean_1, data.max(), 0.001).reshape(-1,1)
            probs = model2.predict_proba(vals)
            argprob = next(filter( lambda x: probs[x][1] > 0.5, np.arange(0, len(probs),1)))
            prob = vals[argprob]
            print(probs[argprob])
            print(argprob)
            df.loc[i] = [int(i+1), mean_1, mean_2, var_1, var_2, prob[0]]
        elif mean_2 < mean_1:
            print([mean_1, mean_2])
            vals = np.arange(mean_2, data.max(), 0.001).reshape(-1,1)
            probs = model2.predict_proba(vals)
            argprob = next(filter( lambda x: probs[x][0] > 0.5, np.arange(0, len(probs),1)))
            prob = vals[argprob]
            print(probs[argprob])
            print(argprob)
            df.loc[i] = [int(i+1), mean_2, mean_1, var_2, var_1, prob[0]]

df.to_csv("wm-pvc-moments-prob.csv")