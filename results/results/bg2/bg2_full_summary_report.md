# Sweep Summary (monitor: `de00`, mode: `min`)

Total unique runs: **48**
Best run metric (val): **1.7373**

## Constants
| included_folders   |   epochs |   mse_weight_epochs | optim   | lr_auto   | lr_schedule   |   weight_decay |
|:-------------------|---------:|--------------------:|:--------|:----------|:--------------|---------------:|
| focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 |

## Grid keys (vary across runs)
`backbone`, `features`, `model`

## Top Runs
| run_id       | included_folders   |   epochs |   mse_weight_epochs | optim   | lr_auto   | lr_schedule   |   weight_decay | backbone   | features        | model   |   best_metric |   best_epoch |
|:-------------|:-------------------|---------:|--------------------:|:--------|:----------|:--------------|---------------:|:-----------|:----------------|:--------|--------------:|-------------:|
| 6e87c305e854 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+mean+meta | xattn   |       1.73729 |           63 |
| 3f723eb52be0 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+mean+meta | xattn   |       1.73729 |           73 |
| 14159ef67ff2 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+mean      | xattn   |       1.79656 |           67 |
| 2fc7fb93d7b3 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+mean      | xattn   |       1.79656 |           77 |
| 359cb7704fca | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+mean+meta | film    |       1.87576 |           67 |
| 335ad74f72de | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+mean+meta | film    |       1.87576 |           77 |
| caf256c9b8a9 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+mean+meta | film    |       1.88475 |           42 |
| 1dd8968c48bd | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+mean+meta | film    |       1.88475 |           69 |
| 21a0190ee6f1 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image           | film    |       1.89915 |           44 |
| fbb24c7e201f | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image           | film    |       1.89915 |           69 |
| 1f834ddfeef4 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+mean      | xattn   |       1.90767 |           71 |
| b2b2bd1d4bf3 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+mean      | xattn   |       1.90767 |           61 |
| d44325f74475 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+mean      | film    |       1.93406 |           58 |
| 629d07efcde1 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+mean      | film    |       1.93406 |           69 |
| 4a5fd0d3a900 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+meta      | xattn   |       1.96142 |           64 |
| 63de91889891 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+meta      | xattn   |       1.96142 |           74 |
| 45720b8dd8be | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+mean      | film    |       1.97425 |           57 |
| e6be0adecbe8 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+mean      | film    |       1.97425 |           69 |
| 493fa80601ab | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+meta      | film    |       1.97447 |           55 |
| 3b0ca64bc6c1 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+meta      | film    |       1.97447 |           69 |
| 657b7d8e311f | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+meta      | xattn   |       2.03067 |           69 |
| f8d074b9cd9b | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+meta      | xattn   |       2.03485 |           69 |
| 6796011c5534 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+mean+meta | xattn   |       2.05231 |           80 |
| e21aba1e4b1a | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+mean+meta | xattn   |       2.05231 |           70 |
| 09e5536fc9a0 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+mean      | late    |       2.07463 |           61 |

## Grouped by backbone
| backbone   |   count |   median |    mean |       std |
|:-----------|--------:|---------:|--------:|----------:|
| resnet18   |      24 |  2.04358 | 2.02159 | 0.0980694 |
| smallcnn   |      24 |  2.15739 | 2.12563 | 0.251525  |

## Grouped by features
| features        |   count |   median |    mean |      std |
|:----------------|--------:|---------:|--------:|---------:|
| image+mean      |      12 |  1.95415 | 1.98084 | 0.133103 |
| image+mean+meta |      12 |  1.96853 | 1.95783 | 0.14133  |
| image+meta      |      12 |  2.08956 | 2.14549 | 0.186371 |
| image           |      12 |  2.18668 | 2.21027 | 0.20433  |

## Grouped by model
| model   |   count |   median |    mean |      std |
|:--------|--------:|---------:|--------:|---------:|
| film    |      16 |  1.95415 | 2.0118  | 0.163874 |
| xattn   |      16 |  1.99604 | 1.99334 | 0.179447 |
| late    |      16 |  2.13891 | 2.21568 | 0.169935 |

## Grouped by backbone+features
| backbone   | features        |   count |   median |    mean |       std |
|:-----------|:----------------|--------:|---------:|--------:|----------:|
| smallcnn   | image+mean+meta |       6 |  1.88475 | 1.91299 | 0.17118   |
| resnet18   | image+mean      |       6 |  1.93406 | 1.97212 | 0.0802784 |
| smallcnn   | image+mean      |       6 |  1.97425 | 1.98956 | 0.179859  |
| resnet18   | image+meta      |       6 |  2.03276 | 2.0505  | 0.0771786 |
| resnet18   | image+mean+meta |       6 |  2.05231 | 2.00266 | 0.0990746 |
| resnet18   | image           |       6 |  2.13355 | 2.06106 | 0.125639  |
| smallcnn   | image           |       6 |  2.3082  | 2.35948 | 0.150445  |
| smallcnn   | image+meta      |       6 |  2.32908 | 2.24048 | 0.220911  |

## Grouped by backbone+model
| backbone   | model   |   count |   median |    mean |       std |
|:-----------|:--------|--------:|---------:|--------:|----------:|
| smallcnn   | xattn   |       8 |  1.87899 | 1.95087 | 0.237382  |
| resnet18   | film    |       8 |  1.91661 | 1.92086 | 0.0398333 |
| resnet18   | xattn   |       8 |  2.04358 | 2.0358  | 0.0923581 |
| smallcnn   | film    |       8 |  2.09856 | 2.10274 | 0.1925    |
| resnet18   | late    |       8 |  2.10674 | 2.10809 | 0.0332482 |
| smallcnn   | late    |       8 |  2.3144  | 2.32327 | 0.185251  |

## Grouped by features+model
| features        | model   |   count |   median |    mean |       std |
|:----------------|:--------|--------:|---------:|--------:|----------:|
| image+mean      | xattn   |       4 |  1.85211 | 1.85211 | 0.0641475 |
| image+mean+meta | film    |       4 |  1.88025 | 1.88025 | 0.0051897 |
| image+mean+meta | xattn   |       4 |  1.8948  | 1.8948  | 0.181873  |
| image+mean      | film    |       4 |  1.95415 | 1.95415 | 0.0232005 |
| image+meta      | xattn   |       4 |  1.99604 | 1.99709 | 0.0412261 |
| image           | film    |       4 |  2.06102 | 2.06102 | 0.186903  |
| image+mean+meta | late    |       4 |  2.09843 | 2.09843 | 0.0213605 |
| image+mean      | late    |       4 |  2.13625 | 2.13625 | 0.0711459 |
| image+meta      | film    |       4 |  2.15178 | 2.15178 | 0.204738  |
| image           | xattn   |       4 |  2.22934 | 2.22934 | 0.0910634 |
| image+meta      | late    |       4 |  2.28761 | 2.28761 | 0.165518  |
| image           | late    |       4 |  2.34045 | 2.34045 | 0.23891   |
