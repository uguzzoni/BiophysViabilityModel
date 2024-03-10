# BiophysViabilityModel

# Unsupervised modeling of mutational landscapes of adeno-associated viruses viability

## Description

Implementation of a inference method of the genotype–fitness relationship trained on sequencing samples from multiple rounds of a screening experiment.
The biophysically-inspired model is designed to predict the viability of genetic variants in DMS experiments of Adeno-associated viruses 2 (AAV2). 
This model is tailored to a specific segment of the CAP region within AAV2’s capsid protein.
The model comprises three main steps: selection, amplification, and sequencing. During the selection phase, mutated viruses are introduced into tissue cells, where their survival and  replication depend on their ability to form functional capsids. Typically, only a small fraction of the initial population of viruses is selected during this phase. Following selection, viral DNA is extracted from a sample of cells and undergoes amplification. Finally, the amplified DNA is sequenced to gather information about the genetic composition of the selected viruses.

More details are available in [Unsupervised modeling of mutational landscapes of adeno-associated viruses viability](https://www.biorxiv.org/content/10.1101/2023.10.26.564138v1.full.pdf) and [Unsupervised Inference of Protein Fitness Landscape from Deep Mutational Scan](https://academic.oup.com/mbe/article/38/1/318/5889958)
Please cite the first paper if you use (even partially) this code.

## Installation

This code is written in Julia language. To add `BiophysViabilityModel` package on Julia REPL run
```
Import("Pkg");Pkg.add("git@github.com:uguzzoni/BiophysViabilityModel.git")
```

## Quick start Example

```julia

using JLD2, BiophysViabilityModel

data = load("data/data_exp1.jld2")["data_experiment1"]

model = Model(
        (BiophysViabilityModel.ZeroEnergy(), BiophysViabilityModel.Kelsic_model()),
        zeros(2,1),
        zeros(1),
        reshape([false, true], 2, 1),
        reshape([true, false], 2, 1)
    ) 

history = learn!(model, data; epochs = 1:100, batchsize=256)

```

## Contributions

[Jorge Fernandez de Cossio Diaz](https://github.com/cossio),  [Guido Uguzzoni](https://github.com/uguzzoni)([GU](mailto:guido.uguzzoni@gmail.com)), [Matteo de Leonardis](https://github.com/matteodeleonardis)([MdL](mailto:matteo.deleonardis2@gmail.com)),[Andrea Pagnani](https://github.com/pagnani)

## Maintainers
[Guido Uguzzoni](https://github.com/uguzzoni)([GU](mailto:guido.uguzzoni@gmail.com)), [Matteo de Leonardis](https://github.com/matteodeleonardis)([MdL](mailto:matteo.deleonardis2@gmail.com))
If you want to participate write us ([sibyl-team](mailto:sibylteam@gmail.com?subject=[GitHub]%20Source%20sibilla)) or make a pull request.

## License
[MIT license](LICENSE)
