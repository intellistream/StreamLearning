## StreamFP

StreamFP is the implementation repository for fingerprint-guided data selection in efficient stream learning. The canonical maintained repository is [DataSysResearch/StreamFP](https://github.com/DataSysResearch/StreamFP). It was migrated from the historical `intellistream/StreamLearning` repository; that old location is retained only as provenance and may redirect here.

StreamFP belongs to DataSys because its primary concern is online data
selection, replay-buffer maintenance, and update efficiency over evolving data,
not application-level agent or workflow orchestration.

## Project status

This repository is a research artifact for reproducing the StreamFP
experiments. It is not presented as a general-purpose training framework.

## Publication

This repository accompanies the following paper:

- Changwu Li et al. "StreamFP: Fingerprint-guided Data Selection for Efficient
  Stream Learning." The ACM Web Conference 2026 (WWW 2026).
  [DOI](https://doi.org/10.1145/3774904.3792584)

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

## License

No repository-wide license has been declared. Contact the maintainers before
redistributing or reusing the code, and review the terms of bundled or imported
third-party components separately.
