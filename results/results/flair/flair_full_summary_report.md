# Sweep Summary (monitor: `de00`, mode: `min`)

Total unique runs: **12**
Best run metric (val): **2.0866**

## Constants
| included_folders   |   epochs |   mse_weight_epochs | optim   | lr_auto   | lr_schedule   |   weight_decay |
|:-------------------|---------:|--------------------:|:--------|:----------|:--------------|---------------:|
| focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 |

## Grid keys (vary across runs)
`backbone`, `features`, `meta_encoder`, `model`

## Top Runs
| run_id       | included_folders   |   epochs |   mse_weight_epochs | optim   | lr_auto   | lr_schedule   |   weight_decay | backbone   | features        | meta_encoder   | model   |   best_metric |   best_epoch |
|:-------------|:-------------------|---------:|--------------------:|:--------|:----------|:--------------|---------------:|:-----------|:----------------|:---------------|:--------|--------------:|-------------:|
| 4ea18928ca62 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+mean+meta | flair          | film    |       2.08656 |           55 |
| 694c395e66f5 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+meta      | flair          | film    |       2.08656 |           55 |
| fcacf90b26b8 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+mean+meta | flair          | late    |       2.1634  |           32 |
| c429cd01ee31 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+meta      | flair          | late    |       2.1634  |           32 |
| 8cf3a3dc3541 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+mean+meta | flair          | xattn   |       2.27506 |           64 |
| b14ca8d172fa | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+meta      | flair          | xattn   |       2.27506 |           64 |
| 938986b9947c | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+mean+meta | flair          | film    |       2.30761 |           52 |
| f849949b3902 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+meta      | flair          | film    |       2.30761 |           52 |
| 311bdd58d501 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+mean+meta | flair          | xattn   |       2.43408 |           49 |
| 9f63ecad378b | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+meta      | flair          | xattn   |       2.43408 |           49 |
| 6a1d681c114c | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+mean+meta | flair          | late    |       2.46937 |           54 |
| 07d387471daa | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+meta      | flair          | late    |       2.46937 |           54 |

## Grouped by backbone
| backbone   |   count |   median |    mean |       std |
|:-----------|--------:|---------:|--------:|----------:|
| resnet18   |       6 |  2.1634  | 2.22801 | 0.163275  |
| smallcnn   |       6 |  2.30761 | 2.35068 | 0.0930815 |

## Grouped by features
| features        |   count |   median |    mean |      std |
|:----------------|--------:|---------:|--------:|---------:|
| image+mean+meta |       6 |  2.29133 | 2.28935 | 0.148914 |
| image+meta      |       6 |  2.29133 | 2.28935 | 0.148914 |

## Grouped by meta_encoder
| meta_encoder   |   count |   median |    mean |      std |
|:---------------|--------:|---------:|--------:|---------:|
| flair          |      12 |  2.29133 | 2.28935 | 0.141984 |

## Grouped by model
| model   |   count |   median |    mean |       std |
|:--------|--------:|---------:|--------:|----------:|
| film    |       4 |  2.19708 | 2.19708 | 0.127622  |
| late    |       4 |  2.31638 | 2.31638 | 0.17665   |
| xattn   |       4 |  2.35457 | 2.35457 | 0.0918102 |

## Grouped by backbone+features
| backbone   | features        |   count |   median |    mean |      std |
|:-----------|:----------------|--------:|---------:|--------:|---------:|
| resnet18   | image+mean+meta |       3 |  2.1634  | 2.22801 | 0.182547 |
| resnet18   | image+meta      |       3 |  2.1634  | 2.22801 | 0.182547 |
| smallcnn   | image+mean+meta |       3 |  2.30761 | 2.35068 | 0.104068 |
| smallcnn   | image+meta      |       3 |  2.30761 | 2.35068 | 0.104068 |

## Grouped by backbone+meta_encoder
| backbone   | meta_encoder   |   count |   median |    mean |       std |
|:-----------|:---------------|--------:|---------:|--------:|----------:|
| resnet18   | flair          |       6 |  2.1634  | 2.22801 | 0.163275  |
| smallcnn   | flair          |       6 |  2.30761 | 2.35068 | 0.0930815 |

## Grouped by backbone+model
| backbone   | model   |   count |   median |    mean |   std |
|:-----------|:--------|--------:|---------:|--------:|------:|
| resnet18   | film    |       2 |  2.08656 | 2.08656 |     0 |
| resnet18   | late    |       2 |  2.1634  | 2.1634  |     0 |
| smallcnn   | xattn   |       2 |  2.27506 | 2.27506 |     0 |
| smallcnn   | film    |       2 |  2.30761 | 2.30761 |     0 |
| resnet18   | xattn   |       2 |  2.43408 | 2.43408 |     0 |
| smallcnn   | late    |       2 |  2.46937 | 2.46937 |     0 |

## Grouped by features+meta_encoder
| features        | meta_encoder   |   count |   median |    mean |      std |
|:----------------|:---------------|--------:|---------:|--------:|---------:|
| image+mean+meta | flair          |       6 |  2.29133 | 2.28935 | 0.148914 |
| image+meta      | flair          |       6 |  2.29133 | 2.28935 | 0.148914 |

## Grouped by features+model
| features        | model   |   count |   median |    mean |      std |
|:----------------|:--------|--------:|---------:|--------:|---------:|
| image+mean+meta | film    |       2 |  2.19708 | 2.19708 | 0.156305 |
| image+meta      | film    |       2 |  2.19708 | 2.19708 | 0.156305 |
| image+mean+meta | late    |       2 |  2.31638 | 2.31638 | 0.216352 |
| image+meta      | late    |       2 |  2.31638 | 2.31638 | 0.216352 |
| image+mean+meta | xattn   |       2 |  2.35457 | 2.35457 | 0.112444 |
| image+meta      | xattn   |       2 |  2.35457 | 2.35457 | 0.112444 |

## Grouped by meta_encoder+model
| meta_encoder   | model   |   count |   median |    mean |       std |
|:---------------|:--------|--------:|---------:|--------:|----------:|
| flair          | film    |       4 |  2.19708 | 2.19708 | 0.127622  |
| flair          | late    |       4 |  2.31638 | 2.31638 | 0.17665   |
| flair          | xattn   |       4 |  2.35457 | 2.35457 | 0.0918102 |
