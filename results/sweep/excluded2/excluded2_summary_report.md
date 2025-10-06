# Sweep Summary (monitor: `de00`, mode: `min`)

Total unique runs: **111**
Best run metric: **2.2008**

## Top Runs
| run_id       |   best_metric |   best_epoch | model   | backbone   | features        | excluded_folders   |   mse_weight_epochs |   mse_weight_start |    lr |   weight_decay | pred_activation   |   activation_eps |   seed |
|:-------------|--------------:|-------------:|:--------|:-----------|:----------------|:-------------------|--------------------:|-------------------:|------:|---------------:|:------------------|-----------------:|-------:|
| 44d523e40975 |       2.20083 |           84 | xattn   | smallcnn   | image+meta      | pixel+samsung      |                  25 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 618959f960f8 |       2.26554 |           98 | xattn   | smallcnn   | image+meta      | pixel+samsung      |                  50 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 5b134bc1d2a0 |       2.32942 |           73 | xattn   | smallcnn   | image+mean+meta | pixel+samsung      |                  10 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| dc8d5d6de90b |       2.36066 |           84 | late    | smallcnn   | image+mean+meta | pixel+samsung      |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 9421e6753219 |       2.37014 |           65 | xattn   | smallcnn   | image+mean+meta | pixel+samsung      |                  50 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 0dc7f179de62 |       2.42097 |           89 | xattn   | smallcnn   | image+meta      | samsung            |                  10 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 60b1d28f76a2 |       2.46692 |           51 | xattn   | smallcnn   | image+meta      | pixel+samsung      |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 2bfd0005e01b |       2.53366 |           67 | late    | smallcnn   | image+meta      | pixel+samsung      |                  50 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| db52d586050b |       2.54474 |           35 | xattn   | smallcnn   | image+mean+meta | pixel+samsung      |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 8edd934ff9ef |       2.56839 |           55 | late    | smallcnn   | image+mean+meta | pixel+samsung      |                  50 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 547c156fe365 |       2.56854 |           54 | late    | smallcnn   | image+mean+meta | pixel+samsung      |                  25 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| cfd3a6478b46 |       2.61188 |           46 | late    | smallcnn   | image+mean+meta | pixel+samsung      |                  10 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 87ca9ec2e4bb |       2.61548 |           59 | late    | resnet18   | image+mean+meta | pixel+samsung      |                  50 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| ed864b7cf1b8 |       2.61622 |           54 | xattn   | resnet18   | image+meta      | pixel+samsung      |                  50 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| ab7ff70c2453 |       2.61819 |           78 | xattn   | smallcnn   | image+meta      | pixel              |                  10 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 9d5cd27a1b2b |       2.61957 |           48 | late    | smallcnn   | image+meta      | pixel+samsung      |                  10 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 5e73bba08af7 |       2.62482 |           72 | xattn   | smallcnn   | image+mean+meta | samsung            |                  10 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 7137ce7c96bf |       2.62598 |           95 | xattn   | smallcnn   | image+mean+meta | pixel              |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| f88f936814bb |       2.64301 |           38 | xattn   | smallcnn   | image+mean+meta | pixel+samsung      |                  25 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 459294bacc0f |       2.65717 |           72 | xattn   | resnet18   | image+meta      | samsung            |                  10 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 5e5d47c06fe3 |       2.66324 |           84 | late    | smallcnn   | image+mean+meta | samsung            |                  50 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 0857967e35c3 |       2.6636  |           93 | xattn   | smallcnn   | image+meta      | pixel              |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| a23566bbeab9 |       2.66518 |           91 | late    | smallcnn   | image+meta      | samsung            |                  25 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 23b64a4fdfce |       2.66722 |           84 | xattn   | smallcnn   | image+mean+meta | samsung            |                  50 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 06b1ca317344 |       2.71111 |           92 | late    | smallcnn   | image+meta      | samsung            |                  10 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |

## Grouped by model
| model   |   count |   median |    mean |      std |
|:--------|--------:|---------:|--------:|---------:|
| xattn   |      47 |  2.87128 | 2.90456 | 0.379489 |
| late    |      64 |  3.08064 | 3.07095 | 0.326785 |

## Grouped by backbone
| backbone   |   count |   median |    mean |      std |
|:-----------|--------:|---------:|--------:|---------:|
| smallcnn   |      64 |  2.87577 | 2.88814 | 0.34545  |
| resnet18   |      47 |  3.14132 | 3.15351 | 0.318831 |

## Grouped by features
| features        |   count |   median |    mean |      std |
|:----------------|--------:|---------:|--------:|---------:|
| image+mean+meta |      48 |  2.91991 | 2.92216 | 0.298407 |
| image+meta      |      63 |  2.98283 | 3.06019 | 0.38934  |

## Grouped by mse_weight_epochs
|   mse_weight_epochs |   count |   median |    mean |      std |
|--------------------:|--------:|---------:|--------:|---------:|
|                  50 |      28 |  2.91508 | 2.95925 | 0.370353 |
|                  25 |      28 |  2.93717 | 3.02275 | 0.369125 |
|                  75 |      27 |  2.95489 | 3.00394 | 0.341193 |
|                  10 |      28 |  3.01765 | 3.01618 | 0.366409 |

## Grouped by excluded_folders
| excluded_folders   |   count |   median |    mean |      std |
|:-------------------|--------:|---------:|--------:|---------:|
| pixel+samsung      |      28 |  2.63129 | 2.71972 | 0.312215 |
| samsung            |      28 |  2.89431 | 2.92809 | 0.272468 |
| pixel              |      27 |  3.10085 | 3.0712  | 0.279768 |
| nan                |      28 |  3.28334 | 3.28552 | 0.314305 |

### Elimination candidates (excluded_folders)
|   excluded_folders |   count |   median |    mean |      std |   delta_vs_best |
|-------------------:|--------:|---------:|--------:|---------:|----------------:|
|                nan |      28 |  3.28334 | 3.28552 | 0.314305 |        0.652056 |

## Grouped by model+backbone
| model   | backbone   |   count |   median |    mean |      std |
|:--------|:-----------|--------:|---------:|--------:|---------:|
| xattn   | smallcnn   |      32 |  2.77625 | 2.79389 | 0.333335 |
| late    | smallcnn   |      32 |  2.98567 | 2.98239 | 0.336256 |
| late    | resnet18   |      32 |  3.14009 | 3.15952 | 0.296244 |
| xattn   | resnet18   |      15 |  3.23248 | 3.14067 | 0.373388 |

## Grouped by model+features
| model   | features        |   count |   median |    mean |      std |
|:--------|:----------------|--------:|---------:|--------:|---------:|
| xattn   | image+mean+meta |      16 |  2.75702 | 2.72059 | 0.196754 |
| xattn   | image+meta      |      31 |  2.92867 | 2.99952 | 0.417245 |
| late    | image+meta      |      32 |  3.04925 | 3.11896 | 0.356997 |
| late    | image+mean+meta |      32 |  3.09079 | 3.02295 | 0.291234 |

## Grouped by model+mse_weight_epochs
| model   |   mse_weight_epochs |   count |   median |    mean |      std |
|:--------|--------------------:|--------:|---------:|--------:|---------:|
| xattn   |                  25 |      12 |  2.77942 | 2.94968 | 0.426169 |
| xattn   |                  50 |      12 |  2.88101 | 2.85949 | 0.378852 |
| xattn   |                  10 |      12 |  2.88192 | 2.87159 | 0.363102 |
| xattn   |                  75 |      11 |  2.91219 | 2.94048 | 0.390076 |
| late    |                  75 |      16 |  3.02641 | 3.04757 | 0.308801 |
| late    |                  50 |      16 |  3.05629 | 3.03407 | 0.357357 |
| late    |                  25 |      16 |  3.08183 | 3.07755 | 0.32341  |
| late    |                  10 |      16 |  3.11448 | 3.12463 | 0.340149 |

## Grouped by model+excluded_folders
| model   | excluded_folders   |   count |   median |    mean |      std |
|:--------|:-------------------|--------:|---------:|--------:|---------:|
| xattn   | pixel+samsung      |      12 |  2.58048 | 2.66545 | 0.412587 |
| xattn   | pixel              |      11 |  2.77435 | 2.97196 | 0.378302 |
| xattn   | samsung            |      12 |  2.78062 | 2.90685 | 0.350491 |
| late    | pixel+samsung      |      16 |  2.79347 | 2.76042 | 0.215649 |
| xattn   | nan                |      12 |  2.92641 | 3.0796  | 0.280559 |
| late    | samsung            |      16 |  2.93754 | 2.94402 | 0.207153 |
| late    | pixel              |      16 |  3.1577  | 3.13943 | 0.167547 |
| late    | nan                |      16 |  3.35591 | 3.43995 | 0.246443 |

## Grouped by backbone+features
| backbone   | features        |   count |   median |    mean |      std |
|:-----------|:----------------|--------:|---------:|--------:|---------:|
| smallcnn   | image+mean+meta |      32 |  2.78062 | 2.84843 | 0.310104 |
| smallcnn   | image+meta      |      32 |  2.9065  | 2.92784 | 0.378287 |
| resnet18   | image+mean+meta |      16 |  3.13028 | 3.06962 | 0.213269 |
| resnet18   | image+meta      |      31 |  3.19432 | 3.1968  | 0.356982 |

## Grouped by backbone+mse_weight_epochs
| backbone   |   mse_weight_epochs |   count |   median |    mean |      std |
|:-----------|--------------------:|--------:|---------:|--------:|---------:|
| smallcnn   |                  25 |      16 |  2.7935  | 2.8939  | 0.351112 |
| smallcnn   |                  50 |      16 |  2.81692 | 2.84275 | 0.357749 |
| smallcnn   |                  10 |      16 |  2.89824 | 2.89987 | 0.376553 |
| smallcnn   |                  75 |      16 |  2.95375 | 2.91603 | 0.32395  |
| resnet18   |                  75 |      11 |  2.95489 | 3.13182 | 0.338975 |
| resnet18   |                  50 |      12 |  3.13151 | 3.11457 | 0.340713 |
| resnet18   |                  25 |      12 |  3.16827 | 3.19455 | 0.331635 |
| resnet18   |                  10 |      12 |  3.18567 | 3.17127 | 0.30044  |

## Grouped by backbone+excluded_folders
| backbone   | excluded_folders   |   count |   median |    mean |      std |
|:-----------|:-------------------|--------:|---------:|--------:|---------:|
| smallcnn   | pixel+samsung      |      16 |  2.55656 | 2.57897 | 0.289472 |
| smallcnn   | samsung            |      16 |  2.75768 | 2.83659 | 0.275394 |
| resnet18   | pixel+samsung      |      12 |  2.85624 | 2.90738 | 0.24035  |
| smallcnn   | pixel              |      16 |  2.97356 | 2.97193 | 0.262255 |
| resnet18   | samsung            |      12 |  3.03572 | 3.05008 | 0.224478 |
| smallcnn   | nan                |      16 |  3.13621 | 3.16505 | 0.280575 |
| resnet18   | pixel              |      11 |  3.19432 | 3.2156  | 0.248263 |
| resnet18   | nan                |      12 |  3.42152 | 3.44614 | 0.293046 |

## Grouped by features+mse_weight_epochs
| features        |   mse_weight_epochs |   count |   median |    mean |      std |
|:----------------|--------------------:|--------:|---------:|--------:|---------:|
| image+mean+meta |                  50 |      12 |  2.85099 | 2.92169 | 0.375585 |
| image+mean+meta |                  75 |      12 |  2.90423 | 2.86701 | 0.264379 |
| image+mean+meta |                  25 |      12 |  2.93717 | 2.94589 | 0.25678  |
| image+meta      |                  50 |      16 |  2.94268 | 2.98742 | 0.376122 |
| image+meta      |                  25 |      16 |  2.97758 | 3.08039 | 0.434319 |
| image+mean+meta |                  10 |      12 |  3.01765 | 2.95406 | 0.3139   |
| image+meta      |                  10 |      16 |  3.02235 | 3.06278 | 0.404966 |
| image+meta      |                  75 |      15 |  3.0357  | 3.11349 | 0.363795 |

## Grouped by features+excluded_folders
| features        | excluded_folders   |   count |   median |    mean |      std |
|:----------------|:-------------------|--------:|---------:|--------:|---------:|
| image+mean+meta | pixel+samsung      |      12 |  2.59021 | 2.59965 | 0.194684 |
| image+meta      | pixel+samsung      |      16 |  2.81677 | 2.80977 | 0.357046 |
| image+mean+meta | samsung            |      12 |  2.83968 | 2.90309 | 0.22876  |
| image+meta      | samsung            |      16 |  2.89744 | 2.94683 | 0.307212 |
| image+mean+meta | pixel              |      12 |  3.07182 | 3.0295  | 0.224262 |
| image+meta      | pixel              |      15 |  3.17407 | 3.10457 | 0.321184 |
| image+mean+meta | nan                |      12 |  3.18321 | 3.15641 | 0.231624 |
| image+meta      | nan                |      16 |  3.39892 | 3.38235 | 0.339319 |

## Grouped by mse_weight_epochs+excluded_folders
|   mse_weight_epochs | excluded_folders   |   count |   median |    mean |      std |
|--------------------:|:-------------------|--------:|---------:|--------:|---------:|
|                  50 | pixel+samsung      |       7 |  2.56839 | 2.55033 | 0.197204 |
|                  10 | samsung            |       7 |  2.71111 | 2.86632 | 0.346904 |
|                  25 | pixel+samsung      |       7 |  2.77576 | 2.71919 | 0.286758 |
|                  75 | pixel+samsung      |       7 |  2.78303 | 2.70705 | 0.252299 |
|                  25 | samsung            |       7 |  2.78309 | 2.9005  | 0.239973 |
|                  10 | pixel+samsung      |       7 |  2.82101 | 2.9023  | 0.42582  |
|                  50 | samsung            |       7 |  2.90254 | 2.86818 | 0.17808  |
|                  75 | samsung            |       7 |  2.97882 | 3.07734 | 0.298191 |
|                  10 | pixel              |       7 |  3.04279 | 3.04131 | 0.226227 |
|                  75 | pixel              |       6 |  3.06448 | 2.9809  | 0.280066 |
|                  25 | pixel              |       7 |  3.10085 | 3.1198  | 0.344777 |
|                  50 | pixel              |       7 |  3.14132 | 3.1299  | 0.296288 |
|                  75 | nan                |       7 |  3.21606 | 3.24719 | 0.328387 |
|                  10 | nan                |       7 |  3.30479 | 3.2548  | 0.377901 |
|                  25 | nan                |       7 |  3.32522 | 3.3515  | 0.304025 |
|                  50 | nan                |       7 |  3.34302 | 3.28859 | 0.305878 |
