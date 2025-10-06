# Sweep Summary (monitor: `de00`, mode: `min`)

Total unique runs: **18**
Best run metric (test): **0.8064**

## Constants
| included_folders   |   epochs |   mse_weight_epochs | optim   | lr_auto   | lr_schedule   |   weight_decay |
|:-------------------|---------:|--------------------:|:--------|:----------|:--------------|---------------:|
| focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 |

## Grid keys (vary across runs)
`backbone`, `features`, `model`

## Top Runs
| run_id       | included_folders   |   epochs |   mse_weight_epochs | optim   | lr_auto   | lr_schedule   |   weight_decay | backbone   | features        | model   |   best_metric |   best_epoch |
|:-------------|:-------------------|---------:|--------------------:|:--------|:----------|:--------------|---------------:|:-----------|:----------------|:--------|--------------:|-------------:|
| 02d8e7e08af8 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+mean+meta | xattn   |      0.806383 |           70 |
| fbb44905c3fa | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+mean      | film    |      0.851027 |           58 |
| 00eafd0a5a3a | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+mean      | late    |      0.876823 |           61 |
| 21f6f6b9e4e1 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+mean+meta | film    |      0.907599 |           67 |
| b1e23c379a61 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+mean+meta | late    |      0.949545 |           69 |
| 1e7343f32934 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image           | late    |      1.08005  |           55 |
| ea28a694330a | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+meta      | xattn   |      1.12906  |           69 |
| d5f3f073a8f9 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+meta      | late    |      1.15573  |           56 |
| 244efe38363a | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image           | film    |      1.27167  |           44 |
| c7a0198aa8f8 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+mean+meta | xattn   |      1.31953  |           63 |
| 5ee25539b142 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+meta      | film    |      1.32112  |           55 |
| a140944a9a99 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+mean      | xattn   |      1.33009  |           67 |
| bf520e38f6bc | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+meta      | xattn   |      1.44502  |           64 |
| d40f1514f0fd | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+mean      | film    |      1.74506  |           57 |
| 588c9b7ceb9d | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+mean+meta | film    |      1.78097  |           42 |
| ed29b0902b63 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+mean+meta | late    |      1.79742  |           57 |
| 2edefebbb2bd | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+mean      | late    |      1.86084  |           80 |
| 262e5de002df | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image           | film    |      2.15208  |           57 |

## Grouped by backbone
| backbone   |   count |   median |    mean |      std |
|:-----------|--------:|---------:|--------:|---------:|
| resnet18   |      10 |  1.0148  | 1.0349  | 0.181964 |
| smallcnn   |       8 |  1.76301 | 1.67888 | 0.290654 |

### Elimination candidates (backbone)
| backbone   |   count |   median |    mean |      std |   delta_vs_best |
|:-----------|--------:|---------:|--------:|---------:|----------------:|
| smallcnn   |       8 |  1.76301 | 1.67888 | 0.290654 |        0.748217 |

## Grouped by features
| features        |   count |   median |    mean |      std |
|:----------------|--------:|---------:|--------:|---------:|
| image+mean+meta |       6 |  1.13454 | 1.26024 | 0.444997 |
| image+meta      |       4 |  1.23843 | 1.26273 | 0.148273 |
| image           |       3 |  1.27167 | 1.50127 | 0.571705 |
| image+mean      |       5 |  1.33009 | 1.33277 | 0.471383 |

## Grouped by model
| model   |   count |   median |    mean |      std |
|:--------|--------:|---------:|--------:|---------:|
| late    |       6 |  1.11789 | 1.28673 | 0.431741 |
| xattn   |       5 |  1.31953 | 1.20602 | 0.250515 |
| film    |       7 |  1.32112 | 1.43279 | 0.481113 |

## Grouped by backbone+features
| backbone   | features        |   count |   median |     mean |         std |
|:-----------|:----------------|--------:|---------:|---------:|------------:|
| resnet18   | image+mean      |       2 | 0.863925 | 0.863925 |   0.0182409 |
| resnet18   | image+mean+meta |       3 | 0.907599 | 0.887842 |   0.0735975 |
| resnet18   | image+meta      |       3 | 1.15573  | 1.20197  |   0.104046  |
| resnet18   | image           |       2 | 1.17586  | 1.17586  |   0.135498  |
| smallcnn   | image+meta      |       1 | 1.44502  | 1.44502  | nan         |
| smallcnn   | image+mean      |       3 | 1.74506  | 1.64533  |   0.279075  |
| smallcnn   | image+mean+meta |       3 | 1.78097  | 1.63264  |   0.271286  |
| smallcnn   | image           |       1 | 2.15208  | 2.15208  | nan         |

## Grouped by backbone+model
| backbone   | model   |   count |   median |     mean |       std |
|:-----------|:--------|--------:|---------:|---------:|----------:|
| resnet18   | xattn   |       2 | 0.967723 | 0.967723 | 0.228169  |
| resnet18   | late    |       4 | 1.0148   | 1.01554  | 0.125715  |
| resnet18   | film    |       4 | 1.08964  | 1.08786  | 0.24275   |
| smallcnn   | xattn   |       3 | 1.33009  | 1.36488  | 0.0696034 |
| smallcnn   | film    |       3 | 1.78097  | 1.8927   | 0.225342  |
| smallcnn   | late    |       2 | 1.82913  | 1.82913  | 0.0448398 |

## Grouped by features+model
| features        | model   |   count |   median |    mean |        std |
|:----------------|:--------|--------:|---------:|--------:|-----------:|
| image+mean+meta | xattn   |       2 |  1.06296 | 1.06296 |   0.362849 |
| image           | late    |       1 |  1.08005 | 1.08005 | nan        |
| image+meta      | late    |       1 |  1.15573 | 1.15573 | nan        |
| image+meta      | xattn   |       2 |  1.28704 | 1.28704 |   0.223415 |
| image+mean      | film    |       2 |  1.29804 | 1.29804 |   0.632178 |
| image+meta      | film    |       1 |  1.32112 | 1.32112 | nan        |
| image+mean      | xattn   |       1 |  1.33009 | 1.33009 | nan        |
| image+mean+meta | film    |       2 |  1.34428 | 1.34428 |   0.617564 |
| image+mean      | late    |       2 |  1.36883 | 1.36883 |   0.695802 |
| image+mean+meta | late    |       2 |  1.37348 | 1.37348 |   0.59954  |
| image           | film    |       2 |  1.71187 | 1.71187 |   0.622541 |
