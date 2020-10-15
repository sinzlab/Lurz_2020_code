# Generalization in data-driven models of primary visual cortex (Code)
Code base for "Generalization in data-driven models of primary visual cortex", Lurz et al. 2020

## Requirements

* `docker` and `docker-compose`
* [GIN](https://web.gin.g-node.org/G-Node/Info/wiki/GinCli#quickstart) along with `git` and `git-annex` to download the data. 

## Data License

The data shared with this code is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>. This license requires that you contact us before you use the data in your own research. In particular, this means that you have to ask for permission if you intend to publish a new analysis performed with this data (no derivative works-clause).

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a>

## Quickstart

Go to a folder of your choice and type the following commands in a [shell of your choice](https://fishshell.com/):

```bash
git clone https://github.com/sinzlab/Lurz_2020_code.git

# get the data
cd Lurz_2020_code/notebooks/data
gin get cajal/Lurz2020 # might take a while; fast internet recommended
cd -

# create docker container (possibly you need sudo)
cd Lurz_2020_code/
docker-compose run notebook
```

Now you should be able to access the jupyter notebooks via `YOURCOMPUTER:8888` in the browser. 
The data you downloaded is the test animal (Figure 5 in the paper) that we tested our transfer cores on. Our best transfer core (11-S in the paper, orange line) is in `Lurz_2020_core/notebooks/models` and can be loaded as described in `Lurz_2020_core/notebooks/example.ipynb`.

If you want to predict your own data with our core, copy the data to the folder `Lurz_2020_core/notebooks/data`.
