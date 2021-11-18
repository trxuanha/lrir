# Latent Representation Intervention Recommendation
A python implementation of Latent Representation Intervention Recommendation (MCT) in paper "What is the Most Effective Intervention to Increase Employment Retention for this Disabled Worker?". This implementation also uses R packages to execute baseline methods and use survival analysis.

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

**1. Reproduce results in the paper with existing data**

    Run shell script "generate_results"

**2. Reproduce results in the paper from scratch**

    Run shell scripts: do_exp_ACTG175, do_exp_GBSG, do_exp_Hr, do_exp_Turnover, generate_results.
