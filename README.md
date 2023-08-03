# Horvath's Clock

Replication of [Horvath's Clock](https://en.wikipedia.org/wiki/Epigenetic_clock), regressing the age from methylation data over 450k wide methylation vectors of ~8500 patients.

Uses data from the [EWAS Data Hub ](https://academic.oup.com/nar/article/48/D1/D890/5580903).

We find a median absolute error of ~2.5 years and a R**2 of score of 94, using an elastic net model, like in the original paper.

Download the data from: https://download.cncb.ac.cn/ewas/datahub/download/age_methylation_v1.zip

The data needs to be extracted, shuffled & split. See scripts.

The data is filtered to keep only the columns that have 10 or less missing values. This leaves us with ~ 125k wide vectors.

The data is then imputed using the mean method (missing values are replaced by their average). Imputation is required by linear methods.

The features columns are then selected in decreasing order of their F Statistic. We get very good results with the most correlated 1000 features, & the model finds up to 50%+ sparsity in those, even.

The analysis takes place in `main.ipynb`.


