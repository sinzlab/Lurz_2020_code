from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name="lurz2020",
    version="0.0.0",
    description='Code base for "Generalization in data-driven models of primary visual cortex", Lurz et al. 2020',
    author="Konstantin-Klemens Lurz",
    author_email="konstantin.lurz@uni-tuebingen.de",
    packages=find_packages(exclude=[]),
    install_requires=["neuralpredictors~=0.0.1"],
)
