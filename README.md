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

# Infrastructure used to run experiments:
* OS: Red Hat Enterprise Linux, version 7.8.
* CPU: Intel(R) Xeon(R) Gold 6246 CPU @ 3.30GHz).
* RAM: 16 GB.

# Public datasets

Employee turnover (TO): This dataset contains information from 1129 people. It has 16 features describing the
characteristics of Russian workers at a company. The outcome is time that employees remain working at the company.

Human resource (HR): This dataset contains information on
15000 people with 10 features describing the characteristics
of employees and their employment time at a company.

AIDS Clinical Trials Group Protocol 175 (ACTG175):
This dataset consists of information of 2139 HIV infected
subjects. There are 25 features describing the characteristics
of patients, treatments they received, and outcomes.

German Breast Cancer Study Group (GBSG): This dataset
consists of 2232 patients from the study about the effect of
chemotherapy and hormone treatment on the survival rate.
The dataset has nine features describing the characteristics
of patients, treatments they received, and outcomes.

# Usage

**1. Buil the prediction models and perform experiments for four datasets**

    Run shell scripts: do_exp_ACTG175, do_exp_GBSG, do_exp_Hr, do_exp_Turnover.

**2. Reproduce results in the paper**

    Run shell scripts: generate_results.
