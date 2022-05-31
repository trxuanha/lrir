# Latent Representation Intervention Recommendation
A python implementation of Latent Representation Intervention Recommendation (LRIR) in paper "What is the Most Effective Intervention to Increase Job Retention for this Disabled Worker?". This implementation also uses R packages to execute baseline methods and use survival analysis.

# Installation
Installation requirements for Maximum Causal Tree

* Python >= 3.6
* numpy
* pandas
* scipy
* seaborn
* sklearn
* survival
* survminer
* patchwork
* foreach
* doParallel
* randomForestSRC
* causalTree
* grf

# Usage

**Buil the prediction models and perform experiments for four datasets**

    Run shell scripts: do_exp_ACTG175, do_exp_GBSG, do_exp_Hr, do_exp_Turnover.
