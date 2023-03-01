from setuptools import setup
from setuptools import find_namespace_packages

setup(
    name="asreview-ensemble-classifier",
    version="0.0.1",
    description="ASReivew Ensemble classifier extension",
    url="https://github.com/rohitgarud/Asreview-Ensemble-Classifiers",
    author="Rohit Garud",
    author_email="rohit.garuda1992@gmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="systematic review, ASReview",
    packages=find_namespace_packages(include=["asreviewcontrib.*"]),
    python_requires="~=3.6",
    install_requires=["asreview>=1.0"],
    entry_points={
        "asreview.models.classifiers": [
            "ensemble_nb_lr = asreviewcontrib.models.ensemble:EnsembleNBLRClassifier",
            "ensemble_nb_rf = asreviewcontrib.models.ensemble:EnsembleNBRFClassifier",
            "ensemble_lr_rf = asreviewcontrib.models.ensemble:EnsembleLRRFClassifier",
            "ensemble_nb_lr_rf = asreviewcontrib.models.ensemble:EnsembleNBLRRFClassifier",
        ],
        "asreview.models.feature_extraction": [
            # define feature_extraction algorithms
        ],
        "asreview.models.balance": [
            # define balance strategy algorithms
        ],
        "asreview.models.query": [
            # define query strategy algorithms
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/rohitgarud/Asreview-Ensemble-Classifiers/issues",
        "Source": "https://github.com/rohitgarud/Asreview-Ensemble-Classifiers",
    },
)
