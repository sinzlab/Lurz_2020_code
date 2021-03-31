# Generalization in data-driven models of primary visual cortex (Code)
Code base for "Generalization in data-driven models of primary visual cortex", Lurz et al. ICLR 2021 
Paper: https://openreview.net/forum?id=Tp7kI90Htd
Data: https://gin.g-node.org/cajal/Lurz2020

## Requirements

* `docker` and `docker-compose`
* [GIN](https://web.gin.g-node.org/G-Node/Info/wiki/GinCli#quickstart) along with `git` and `git-annex` to download the data. 


## Quickstart

Go to a folder of your choice and type the following commands in a [shell of your choice](https://fishshell.com/):

```bash
git clone https://github.com/sinzlab/Lurz_2020_code.git

# get the data
cd Lurz_2020_code/notebooks/data
gin login
gin get cajal/Lurz2020 # might take a while; fast internet recommended
cd -

# create docker container (possibly you need sudo)
cd Lurz_2020_code/
docker-compose run notebook
```

Now you should be able to access the jupyter notebooks via `YOURCOMPUTER:8888` in the browser. 
The data you downloaded is the evaluation dataset (Figure 1 in the paper, blue) from the test animal that we tested our transfer cores on in Figure 5. The weights from our best transfer core (11-S in Figure 5, orange line) are stored in `Lurz_2020_core/notebooks/models` and can be loaded as described in `Lurz_2020_core/notebooks/example.ipynb`.

If you want to predict your own data with our core, copy your data to the folder `Lurz_2020_core/notebooks/data`.
