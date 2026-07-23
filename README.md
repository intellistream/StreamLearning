## StreamFP

StreamFP is the implementation repository for fingerprint-guided data selection in efficient stream learning. The canonical maintained repository is [DataSysResearch/StreamFP](https://github.com/DataSysResearch/StreamFP). It was migrated from the historical `intellistream/StreamLearning` repository; that old location is retained only as provenance and may redirect here.

## Publication

This repository accompanies the following paper:

- Changwu Li et al. "StreamFP: Fingerprint-guided Data Selection for Efficient Stream Learning." The ACM Web Conference 2026 (WWW 2026).

## Setup
* Install miniconda
* `conda env create -f environment.yml`
* `conda activate sl`
* Install fastmoe library: https://github.com/laekov/fastmoe/blob/master/doc/installation-guide.md


  
## Datasets
 * Create a folder `data/`
 * **Clear10**, **Clear100**: retrieve from: https://clear-benchmark.github.io/
 * **Stream51**: retrieve from: https://github.com/tyler-hayes/Stream-51
 * **CORe50**: `sh core50.sh`


## Training
All commands should be run under the project root directory. **The scripts are set up for 1 GPUs** but can be modified for your hardware.

```bash
sh experiments/clear10.sh
sh experiments/clear100.sh
sh experiments/core50.sh
sh experiments/stream51.sh
```
