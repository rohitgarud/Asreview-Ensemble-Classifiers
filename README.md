# ASReview Ensemble Classifiers Extension

This extension adds a new set of classifiers by ensembling different basic classifiers such as Naive Bayes (NB), Logistic Regression (LR) and Random Forest (RF).


## Getting started

To install this extension, clone the repository to your system and then run the following command from inside the repository.

```bash
pip install .
```

or you can directly install it from GitHub using

```bash
pip install git+https://github.com/rohitgarud/Asreview-Ensemble-Classifiers.git
```

## Usage

There are currently four different ensemble classifiers are available: `ensemble_nb_lr` (NB+LR), `ensemble_nb_rf` (NB+RF), `ensemble_lr_rf` (LR+RF), `ensemble_nb_lr_rf` (NB+LR+RF). Simulations can be performed using the simulation mode from ASReview CLI using:

```bash
asreview simulate example_data_file.csv -m ensemble_nb_lr
```
Also, a comprehensive simulation study can be performed using the  [ASReview Makita Extension](https://github.com/asreview/asreview-makita) (follow the instructions on the extension GitHub page). One example of simulation using is a comparison of NB and Ensemble of NB and LR classifiers can be performed using:
```bash
asreview makita template multiple_models --classifiers nb ensemble_nb_lr --feature_extractors tfidf -f jobs.bat
```

Four different ensemble strategies are available `mean`, `max`, `multiply` and `random`. The default is the `multiply` ensemble strategy. To use other settings, you have to use the Python API 

## License

Apache 2.0 license
