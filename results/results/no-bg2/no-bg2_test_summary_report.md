# Sweep Summary (monitor: `de00`, mode: `min`)

Total unique runs: **17**
Best run metric (test): **1.2809**

## Constants
| included_folders   |   epochs |   mse_weight_epochs | optim   | lr_auto   | lr_schedule   |   weight_decay |
|:-------------------|---------:|--------------------:|:--------|:----------|:--------------|---------------:|
| focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 |

## Grid keys (vary across runs)
`backbone`, `features`, `model`

## Top Runs
| run_id       | included_folders   |   epochs |   mse_weight_epochs | optim   | lr_auto   | lr_schedule   |   weight_decay | backbone   | features        | model   |   best_metric |   best_epoch |
|:-------------|:-------------------|---------:|--------------------:|:--------|:----------|:--------------|---------------:|:-----------|:----------------|:--------|--------------:|-------------:|
| d5f3f073a8f9 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+meta      | late    |       1.28085 |           55 |
| 21f6f6b9e4e1 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+mean+meta | film    |       1.3111  |           57 |
| 4f8e91e023f8 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+mean      | xattn   |       1.39456 |           39 |
| 1e7343f32934 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image           | late    |       1.4858  |           54 |
| b1e23c379a61 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+mean+meta | late    |       1.51752 |           35 |
| ea28a694330a | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+meta      | xattn   |       1.60373 |           28 |
| 02d8e7e08af8 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+mean+meta | xattn   |       1.60939 |           23 |
| 00eafd0a5a3a | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+mean      | late    |       1.63532 |           34 |
| 9b37d0ef15c1 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image           | xattn   |       1.77706 |           35 |
| fbb44905c3fa | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | resnet18   | image+mean      | film    |       1.77937 |           37 |
| c7a0198aa8f8 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+mean+meta | xattn   |       1.7939  |           50 |
| bf520e38f6bc | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+meta      | xattn   |       1.9983  |           50 |
| ed29b0902b63 | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+mean+meta | late    |       2.00401 |           60 |
| 588c9b7ceb9d | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+mean+meta | film    |       2.01762 |           46 |
| 1f4be4308b5d | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+meta      | film    |       2.08267 |           59 |
| ba3a864d113b | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+meta      | late    |       2.25525 |           48 |
| d40f1514f0fd | focus              |      100 |                  75 | adamw   | False     | cosine        |         0.0001 | smallcnn   | image+mean      | film    |       3.4133  |           56 |

## Grouped by backbone
| backbone   |   count |   median |    mean |      std |
|:-----------|--------:|---------:|--------:|---------:|
| resnet18   |      10 |  1.56063 | 1.53947 | 0.174993 |
| smallcnn   |       7 |  2.01762 | 2.22358 | 0.541958 |

## Grouped by features
| features        |   count |   median |    mean |      std |
|:----------------|--------:|---------:|--------:|---------:|
| image           |       2 |  1.63143 | 1.63143 | 0.20595  |
| image+mean+meta |       6 |  1.70164 | 1.70892 | 0.280852 |
| image+mean      |       4 |  1.70734 | 2.05564 | 0.918926 |
| image+meta      |       5 |  1.9983  | 1.84416 | 0.395209 |

## Grouped by model
| model   |   count |   median |    mean |      std |
|:--------|--------:|---------:|--------:|---------:|
| late    |       6 |  1.57642 | 1.69646 | 0.363233 |
| xattn   |       6 |  1.69322 | 1.69616 | 0.207085 |
| film    |       5 |  2.01762 | 2.12081 | 0.783398 |

## Grouped by backbone+features
| backbone   | features        |   count |   median |    mean |        std |
|:-----------|:----------------|--------:|---------:|--------:|-----------:|
| resnet18   | image+meta      |       2 |  1.44229 | 1.44229 |   0.228312 |
| resnet18   | image+mean+meta |       3 |  1.51752 | 1.47934 |   0.152763 |
| resnet18   | image           |       2 |  1.63143 | 1.63143 |   0.20595  |
| resnet18   | image+mean      |       3 |  1.63532 | 1.60308 |   0.194419 |
| smallcnn   | image+mean+meta |       3 |  2.00401 | 1.93851 |   0.12542  |
| smallcnn   | image+meta      |       3 |  2.08267 | 2.11207 |   0.130973 |
| smallcnn   | image+mean      |       1 |  3.4133  | 3.4133  | nan        |

## Grouped by backbone+model
| backbone   | model   |   count |   median |    mean |      std |
|:-----------|:--------|--------:|---------:|--------:|---------:|
| resnet18   | late    |       4 |  1.50166 | 1.47987 | 0.147451 |
| resnet18   | film    |       2 |  1.54524 | 1.54524 | 0.331113 |
| resnet18   | xattn   |       4 |  1.60656 | 1.59618 | 0.156632 |
| smallcnn   | xattn   |       2 |  1.8961  | 1.8961  | 0.144532 |
| smallcnn   | film    |       3 |  2.08267 | 2.50453 | 0.787691 |
| smallcnn   | late    |       2 |  2.12963 | 2.12963 | 0.177652 |

## Grouped by features+model
| features        | model   |   count |   median |    mean |        std |
|:----------------|:--------|--------:|---------:|--------:|-----------:|
| image+mean      | xattn   |       1 |  1.39456 | 1.39456 | nan        |
| image           | late    |       1 |  1.4858  | 1.4858  | nan        |
| image+mean      | late    |       1 |  1.63532 | 1.63532 | nan        |
| image+mean+meta | film    |       2 |  1.66436 | 1.66436 |   0.49958  |
| image+mean+meta | xattn   |       2 |  1.70164 | 1.70164 |   0.13047  |
| image+mean+meta | late    |       2 |  1.76077 | 1.76077 |   0.343997 |
| image+meta      | late    |       2 |  1.76805 | 1.76805 |   0.689001 |
| image           | xattn   |       1 |  1.77706 | 1.77706 | nan        |
| image+meta      | xattn   |       2 |  1.80102 | 1.80102 |   0.278999 |
| image+meta      | film    |       1 |  2.08267 | 2.08267 | nan        |
| image+mean      | film    |       2 |  2.59634 | 2.59634 |   1.15537  |
