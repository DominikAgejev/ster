# Sweep Summary (monitor: `de00`, mode: `min`)

Total unique runs: **48**
Best run metric (val): **2.0682**

## Constants
| included_folders   |   epochs |   mse_weight_epochs | optim   | lr_auto   | lr_schedule   |   weight_decay |
|:-------------------|---------:|--------------------:|:--------|:----------|:--------------|---------------:|
| focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 |

## Grid keys (vary across runs)
`backbone`, `features`, `model`

## Top Runs
| run_id       | included_folders   |   epochs |   mse_weight_epochs | optim   | lr_auto   | lr_schedule   |   weight_decay | backbone   | features        | model   |   best_metric |   best_epoch |
|:-------------|:-------------------|---------:|--------------------:|:--------|:----------|:--------------|---------------:|:-----------|:----------------|:--------|--------------:|-------------:|
| 89c22698f5c6 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+mean+meta | late    |       2.06815 |           60 |
| dbe2b3762133 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+mean+meta | late    |       2.06815 |           70 |
| caf256c9b8a9 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+mean+meta | film    |       2.14134 |           46 |
| 1dd8968c48bd | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+mean+meta | film    |       2.14134 |           59 |
| adb9c1555e68 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+meta      | film    |       2.19108 |           59 |
| 4c5b00b4cd4f | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+meta      | film    |       2.19108 |           69 |
| 6e87c305e854 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+mean+meta | xattn   |       2.20103 |           50 |
| 3f723eb52be0 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+mean+meta | xattn   |       2.20103 |           60 |
| e2c4f50e3b03 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+meta      | late    |       2.2183  |           48 |
| b4c69706b4dd | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+meta      | late    |       2.2183  |           59 |
| 3b0ca64bc6c1 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+meta      | film    |       2.26008 |           59 |
| 493fa80601ab | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+meta      | film    |       2.26008 |           57 |
| 4a5fd0d3a900 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+meta      | xattn   |       2.35378 |           50 |
| 63de91889891 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+meta      | xattn   |       2.35378 |           60 |
| e21aba1e4b1a | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+mean+meta | xattn   |       2.37695 |           23 |
| 6796011c5534 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+mean+meta | xattn   |       2.37695 |           59 |
| 657b7d8e311f | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+meta      | xattn   |       2.39357 |           28 |
| f8d074b9cd9b | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+meta      | xattn   |       2.39357 |           59 |
| 359cb7704fca | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+mean+meta | film    |       2.43083 |           57 |
| 335ad74f72de | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+mean+meta | film    |       2.43083 |           67 |
| ee46c590a383 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+meta      | late    |       2.51142 |           55 |
| 8d4cb29e2c1e | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+meta      | late    |       2.51142 |           65 |
| fbb24c7e201f | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image           | film    |       2.55697 |           69 |
| 21a0190ee6f1 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image           | film    |       2.55697 |           59 |
| 55fac85d0c33 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+mean+meta | late    |       2.61177 |           35 |

## Grouped by backbone
| backbone   |   count |   median |    mean |      std |
|:-----------|--------:|---------:|--------:|---------:|
| resnet18   |      24 |  2.58437 | 2.72123 | 0.354079 |
| smallcnn   |      24 |  2.7989  | 2.86954 | 0.699329 |

## Grouped by features
| features        |   count |   median |    mean |      std |
|:----------------|--------:|---------:|--------:|---------:|
| image+mean+meta |      12 |  2.28899 | 2.30501 | 0.194844 |
| image+meta      |      12 |  2.30693 | 2.32137 | 0.115789 |
| image+mean      |      12 |  3.25143 | 3.27496 | 0.280581 |
| image           |      12 |  3.38132 | 3.28021 | 0.404799 |

### Elimination candidates (features)
| features   |   count |   median |    mean |      std |   delta_vs_best |
|:-----------|--------:|---------:|--------:|---------:|----------------:|
| image      |      12 |  3.38132 | 3.28021 | 0.404799 |        1.09234  |
| image+mean |      12 |  3.25143 | 3.27496 | 0.280581 |        0.962449 |

## Grouped by model
| model   |   count |   median |    mean |      std |
|:--------|--------:|---------:|--------:|---------:|
| film    |      16 |  2.4939  | 2.68429 | 0.549176 |
| xattn   |      16 |  2.74412 | 2.82006 | 0.52851  |
| late    |      16 |  2.87666 | 2.88182 | 0.597517 |

## Grouped by backbone+features
| backbone   | features        |   count |   median |    mean |       std |
|:-----------|:----------------|--------:|---------:|--------:|----------:|
| smallcnn   | image+mean+meta |       6 |  2.14134 | 2.13684 | 0.0595244 |
| smallcnn   | image+meta      |       6 |  2.2183  | 2.25439 | 0.0779459 |
| resnet18   | image+meta      |       6 |  2.39357 | 2.38836 | 0.112477  |
| resnet18   | image+mean+meta |       6 |  2.43083 | 2.47318 | 0.11002   |
| resnet18   | image           |       6 |  3.09468 | 2.94701 | 0.305081  |
| resnet18   | image+mean      |       6 |  3.14156 | 3.07639 | 0.198858  |
| smallcnn   | image+mean      |       6 |  3.49199 | 3.47352 | 0.197537  |
| smallcnn   | image           |       6 |  3.62935 | 3.6134  | 0.0313157 |

## Grouped by backbone+model
| backbone   | model   |   count |   median |    mean |      std |
|:-----------|:--------|--------:|---------:|--------:|---------:|
| resnet18   | film    |       8 |  2.4939  | 2.51916 | 0.221813 |
| resnet18   | xattn   |       8 |  2.74412 | 2.78101 | 0.42765  |
| smallcnn   | xattn   |       8 |  2.7989  | 2.85911 | 0.642012 |
| smallcnn   | film    |       8 |  2.84153 | 2.84941 | 0.731264 |
| resnet18   | late    |       8 |  2.87666 | 2.86354 | 0.325511 |
| smallcnn   | late    |       8 |  2.92383 | 2.90009 | 0.811379 |

## Grouped by features+model
| features        | model   |   count |   median |    mean |        std |
|:----------------|:--------|--------:|---------:|--------:|-----------:|
| image+meta      | film    |       4 |  2.22558 | 2.22558 | 0.03984    |
| image+mean+meta | film    |       4 |  2.28609 | 2.28609 | 0.167134   |
| image+mean+meta | xattn   |       4 |  2.28899 | 2.28899 | 0.101566   |
| image+mean+meta | late    |       4 |  2.33996 | 2.33996 | 0.313854   |
| image+meta      | late    |       4 |  2.36486 | 2.36486 | 0.169233   |
| image+meta      | xattn   |       4 |  2.37367 | 2.37367 | 0.0229718  |
| image           | film    |       4 |  3.06511 | 3.06511 | 0.58675    |
| image+mean      | film    |       4 |  3.16037 | 3.16037 | 0.382917   |
| image+mean      | xattn   |       4 |  3.25143 | 3.25143 | 0.00856039 |
| image           | xattn   |       4 |  3.36614 | 3.36614 | 0.313454   |
| image           | late    |       4 |  3.40937 | 3.40937 | 0.254009   |
| image+mean      | late    |       4 |  3.41306 | 3.41306 | 0.313507   |
