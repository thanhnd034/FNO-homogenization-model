# Saved notebook evaluations (not a controlled benchmark)

The following numbers are extracted from printed outputs in the committed notebooks. All three notebooks are configured to **load existing trained weights**, skip training, and evaluate against their FFT reference solver.

| Notebook | Native resolution | Test sample count | Mean relative L2 error at native resolution | Mean relative L2 error at 256×256 | Parameters printed |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Baseline-FNO.ipynb](../Baseline-FNO.ipynb) | 32×32 | 16 | 0.9110% | 0.3246% | 3,149,090 |
| [Baseline-UFNO.ipynb](../Baseline-UFNO.ipynb) | 64×64 | 96 | 0.4185% | 3.5545% | 3,274,914 |
| [Baseline-FNO-CNN.ipynb](../Baseline-FNO-CNN.ipynb) | 64×64 | 96 | 0.3930% | 5.1513% | 3,204,578 |

**Important limitations:** These numbers are taken from **different native grids, sample counts and checkpoint configurations** and must not be interpreted as a fair architectural comparison. The checkpoint files (`baseline1_trained.eqx`, `ufno_trained.eqx`, `fno_cnn_trained.eqx`) referred to in the notebooks are not included in the repository. The current cell settings are not necessarily the settings used for initial training. All included figures are exports of saved notebook outputs.

A rigorous comparison should rerun the models at identical data splits, training sample sizes, grids, modes, epochs, seeds, and evaluation metrics.
