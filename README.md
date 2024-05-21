##  StreamPrompt: Learnable Prompt-guided Data Selection for Efficient Stream Learning

## Setup
* Install miniconda
* `conda env create -f environment.yml`
* `conda activate sl`
* Install fastmoe library: https://github.com/laekov/fastmoe/blob/master/doc/installation-guide.md


  
## Datasets
 * Create a folder `data/`
 * **Clear10**, **Clear100**: retrieve from: https://clear-benchmark.github.io/
 * **CORe50**: `sh core50.sh`

## Training
All commands should be run under the project root directory. **The scripts are set up for 1 GPUs** but can be modified for your hardware.

```bash
sh experiments/clear10.sh
sh experiments/imagenet-r.sh
sh experiments/domainnet.sh
```

