# Sweep Summary (monitor: `de00`, mode: `min`)

Total unique runs: **48**
Best run metric: **2.0718**

## Top Runs
| run_id       |   best_metric |   best_epoch | model   | backbone   | features        | included_folders    |   mse_weight_epochs |   mse_weight_start |    lr |   weight_decay | pred_activation   |   activation_eps |   seed |
|:-------------|--------------:|-------------:|:--------|:-----------|:----------------|:--------------------|--------------------:|-------------------:|------:|---------------:|:------------------|-----------------:|-------:|
| 6c765f239d89 |       2.07182 |           62 | xattn   | smallcnn   | image+mean+meta | focus               |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 084d179b431d |       2.09319 |           63 | xattn   | smallcnn   | image+meta      | focus               |                   5 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| ec0120d60609 |       2.15445 |           55 | xattn   | smallcnn   | image+meta      | focus               |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| dd40ac47f6ac |       2.17396 |           44 | late    | resnet18   | image+mean+meta | focus               |                   5 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 6970c6d5e0f9 |       2.2231  |           46 | late    | smallcnn   | image+mean+meta | focus               |                   5 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 5d8913ccb01b |       2.23883 |           74 | film    | resnet18   | image+mean+meta | focus               |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| bf91f303951d |       2.24601 |           46 | late    | smallcnn   | image+mean+meta | focus               |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 79e54edd2212 |       2.26442 |           35 | xattn   | resnet18   | image+mean+meta | focus               |                   5 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 7eb1f72bfdcf |       2.29182 |           28 | late    | resnet18   | image+meta      | focus               |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| b6cf7046eb26 |       2.29532 |           47 | xattn   | resnet18   | image+meta      | focus               |                   5 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 2e1b76a06fb0 |       2.30873 |           23 | xattn   | resnet18   | image+mean+meta | focus               |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 56b34b4aac2b |       2.32083 |           61 | film    | smallcnn   | image+mean+meta | focus               |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 46c2fa48fc19 |       2.32522 |           35 | late    | smallcnn   | image+meta      | focus               |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 76deba5fbea8 |       2.33586 |           18 | late    | resnet18   | image+meta      | focus               |                   5 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 6ef1ecda7952 |       2.36987 |           79 | film    | smallcnn   | image+mean+meta | focus/iphone+iphone |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 18bc43b1bc73 |       2.37267 |           82 | film    | resnet18   | image+meta      | focus/iphone+iphone |                   5 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| b7d5e948e0a9 |       2.40206 |           55 | xattn   | resnet18   | image+meta      | focus/iphone+iphone |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| e732bdb56f9e |       2.4455  |           27 | xattn   | smallcnn   | image+mean+meta | focus               |                   5 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| d19af28c4a1c |       2.45788 |           42 | film    | smallcnn   | image+mean+meta | focus               |                   5 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| c73e6843eeed |       2.46424 |           43 | film    | smallcnn   | image+meta      | focus               |                   5 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| f2bcb5ec4967 |       2.46811 |           58 | film    | smallcnn   | image+mean+meta | focus/iphone+iphone |                   5 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 4562e288adf6 |       2.48731 |           32 | xattn   | resnet18   | image+mean+meta | focus/iphone+iphone |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| ea1c31e23300 |       2.50143 |           25 | late    | resnet18   | image+mean+meta | focus               |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| c330f1845961 |       2.50215 |           55 | xattn   | resnet18   | image+mean+meta | focus/iphone+iphone |                   5 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| dbacf1b12873 |       2.53288 |           37 | late    | resnet18   | image+meta      | focus/iphone+iphone |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |

## Grouped by model
| model   |   count |   median |    mean |      std |
|:--------|--------:|---------:|--------:|---------:|
| xattn   |      16 |  2.4664  | 2.44742 | 0.242586 |
| late    |      16 |  2.58131 | 2.57163 | 0.289793 |
| film    |      16 |  2.5884  | 2.60901 | 0.257964 |

## Grouped by backbone
| backbone   |   count |   median |    mean |      std |
|:-----------|--------:|---------:|--------:|---------:|
| smallcnn   |      24 |  2.50387 | 2.53372 | 0.290062 |
| resnet18   |      24 |  2.51751 | 2.55166 | 0.249401 |

## Grouped by features
| features        |   count |   median |    mean |      std |
|:----------------|--------:|---------:|--------:|---------:|
| image+mean+meta |      24 |  2.47771 | 2.5311  | 0.297638 |
| image+meta      |      24 |  2.59169 | 2.55427 | 0.240074 |

## Grouped by mse_weight_epochs
|   mse_weight_epochs |   count |   median |    mean |      std |
|--------------------:|--------:|---------:|--------:|---------:|
|                  75 |      24 |  2.51715 | 2.52915 | 0.278945 |
|                   5 |      24 |  2.55599 | 2.55622 | 0.261365 |

## Grouped by included_folders
| included_folders    |   count |   median |    mean |      std |
|:--------------------|--------:|---------:|--------:|---------:|
| focus               |      24 |  2.32302 | 2.37514 | 0.186664 |
| focus/iphone+iphone |      24 |  2.69948 | 2.71023 | 0.230387 |

## Grouped by model+backbone
| model   | backbone   |   count |   median |    mean |      std |
|:--------|:-----------|--------:|---------:|--------:|---------:|
| xattn   | resnet18   |       8 |  2.44469 | 2.44476 | 0.157242 |
| film    | smallcnn   |       8 |  2.46618 | 2.54651 | 0.237022 |
| xattn   | smallcnn   |       8 |  2.49256 | 2.45008 | 0.318374 |
| late    | resnet18   |       8 |  2.51715 | 2.5387  | 0.269241 |
| late    | smallcnn   |       8 |  2.64432 | 2.60456 | 0.324018 |
| film    | resnet18   |       8 |  2.68718 | 2.67151 | 0.278369 |

## Grouped by model+features
| model   | features        |   count |   median |    mean |      std |
|:--------|:----------------|--------:|---------:|--------:|---------:|
| film    | image+mean+meta |       8 |  2.463   | 2.55549 | 0.289719 |
| xattn   | image+mean+meta |       8 |  2.4664  | 2.44318 | 0.249482 |
| xattn   | image+meta      |       8 |  2.4914  | 2.45166 | 0.252628 |
| late    | image+meta      |       8 |  2.58131 | 2.54863 | 0.219033 |
| late    | image+mean+meta |       8 |  2.59885 | 2.59463 | 0.361625 |
| film    | image+meta      |       8 |  2.60624 | 2.66254 | 0.22828  |

## Grouped by model+mse_weight_epochs
| model   |   mse_weight_epochs |   count |   median |    mean |      std |
|:--------|--------------------:|--------:|---------:|--------:|---------:|
| xattn   |                  75 |       8 |  2.44469 | 2.41102 | 0.224584 |
| xattn   |                   5 |       8 |  2.47382 | 2.48382 | 0.269511 |
| late    |                  75 |       8 |  2.51715 | 2.52925 | 0.240179 |
| film    |                   5 |       8 |  2.53897 | 2.57083 | 0.153622 |
| film    |                  75 |       8 |  2.5884  | 2.64719 | 0.340096 |
| late    |                   5 |       8 |  2.64432 | 2.614   | 0.343753 |

## Grouped by model+included_folders
| model   | included_folders    |   count |   median |    mean |      std |
|:--------|:--------------------|--------:|---------:|--------:|---------:|
| xattn   | focus               |       8 |  2.27987 | 2.27677 | 0.174456 |
| late    | focus               |       8 |  2.30852 | 2.34454 | 0.160457 |
| film    | focus               |       8 |  2.51919 | 2.50412 | 0.164813 |
| xattn   | focus/iphone+iphone |       8 |  2.58319 | 2.61807 | 0.170605 |
| film    | focus/iphone+iphone |       8 |  2.71468 | 2.71391 | 0.300474 |
| late    | focus/iphone+iphone |       8 |  2.78407 | 2.79872 | 0.190624 |

## Grouped by backbone+features
| backbone   | features        |   count |   median |    mean |      std |
|:-----------|:----------------|--------:|---------:|--------:|---------:|
| smallcnn   | image+mean+meta |      12 |  2.45169 | 2.50995 | 0.320307 |
| resnet18   | image+mean+meta |      12 |  2.50179 | 2.55224 | 0.28576  |
| resnet18   | image+meta      |      12 |  2.55351 | 2.55107 | 0.219992 |
| smallcnn   | image+meta      |      12 |  2.61828 | 2.55748 | 0.268499 |

## Grouped by backbone+mse_weight_epochs
| backbone   |   mse_weight_epochs |   count |   median |    mean |      std |
|:-----------|--------------------:|--------:|---------:|--------:|---------:|
| smallcnn   |                  75 |      12 |  2.45475 | 2.50774 | 0.314059 |
| resnet18   |                  75 |      12 |  2.51715 | 2.55056 | 0.25112  |
| smallcnn   |                   5 |      12 |  2.53897 | 2.55969 | 0.275346 |
| resnet18   |                   5 |      12 |  2.58459 | 2.55275 | 0.258828 |

## Grouped by backbone+included_folders
| backbone   | included_folders    |   count |   median |    mean |      std |
|:-----------|:--------------------|--------:|---------:|--------:|---------:|
| resnet18   | focus               |      12 |  2.32229 | 2.41163 | 0.182896 |
| smallcnn   | focus               |      12 |  2.32302 | 2.33865 | 0.191044 |
| smallcnn   | focus/iphone+iphone |      12 |  2.66621 | 2.72878 | 0.237493 |
| resnet18   | focus/iphone+iphone |      12 |  2.70681 | 2.69168 | 0.232007 |

## Grouped by features+mse_weight_epochs
| features        |   mse_weight_epochs |   count |   median |    mean |      std |
|:----------------|--------------------:|--------:|---------:|--------:|---------:|
| image+mean+meta |                  75 |      12 |  2.42859 | 2.48502 | 0.300447 |
| image+mean+meta |                   5 |      12 |  2.48513 | 2.57718 | 0.300548 |
| image+meta      |                  75 |      12 |  2.57744 | 2.57329 | 0.261106 |
| image+meta      |                   5 |      12 |  2.61828 | 2.53526 | 0.227037 |

## Grouped by features+included_folders
| features        | included_folders    |   count |   median |    mean |      std |
|:----------------|:--------------------|--------:|---------:|--------:|---------:|
| image+mean+meta | focus               |      12 |  2.28658 | 2.32663 | 0.163808 |
| image+meta      | focus               |      12 |  2.40005 | 2.42366 | 0.202203 |
| image+meta      | focus/iphone+iphone |      12 |  2.66621 | 2.68489 | 0.205915 |
| image+mean+meta | focus/iphone+iphone |      12 |  2.75791 | 2.73557 | 0.25919  |

## Grouped by mse_weight_epochs+included_folders
|   mse_weight_epochs | included_folders    |   count |   median |    mean |      std |
|--------------------:|:--------------------|--------:|---------:|--------:|---------:|
|                  75 | focus               |      12 |  2.31478 | 2.35139 | 0.174659 |
|                   5 | focus               |      12 |  2.39068 | 2.39889 | 0.202774 |
|                   5 | focus/iphone+iphone |      12 |  2.67354 | 2.71354 | 0.218437 |
|                  75 | focus/iphone+iphone |      12 |  2.69948 | 2.70692 | 0.251481 |
