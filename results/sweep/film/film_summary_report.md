# Sweep Summary (monitor: `de00`, mode: `min`)

Total unique runs: **62**
Best run metric: **2.0946**

## Top Runs
| run_id       |   best_metric |   best_epoch | model   | backbone   | features        | excluded_folders   |   mse_weight_epochs |   mse_weight_start |    lr |   weight_decay | pred_activation   |   activation_eps |   seed |
|:-------------|--------------:|-------------:|:--------|:-----------|:----------------|:-------------------|--------------------:|-------------------:|------:|---------------:|:------------------|-----------------:|-------:|
| bb13b482261c |       2.09459 |           86 | film    | smallcnn   | image+mean+meta | pixel+samsung      |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 006f08047813 |       2.13844 |           96 | film    | smallcnn   | image+mean+meta | pixel+samsung      |                  10 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| a307aa3356f3 |       2.27352 |           83 | film    | smallcnn   | image+meta      | pixel+samsung      |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 6544bb594985 |       2.34359 |           60 | film    | smallcnn   | image+meta      | pixel+samsung      |                   0 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 61d91036847f |       2.34434 |           95 | late    | smallcnn   | image+meta      | pixel+samsung      |                  25 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 80f82ee029c0 |       2.51441 |           85 | film    | resnet18   | image+mean+meta | samsung            |                  10 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 47f2a37eeaa8 |       2.51877 |           81 | film    | smallcnn   | image+meta      | samsung            |                  10 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| d8e2e6d2ff96 |       2.53988 |           57 | late    | smallcnn   | image+meta      | pixel+samsung      |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 6102c2ec8f4b |       2.54666 |           77 | film    | smallcnn   | image+meta      | pixel+samsung      |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| dc8d5d6de90b |       2.54917 |           53 | late    | smallcnn   | image+mean+meta | pixel+samsung      |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| e0c4639cb20d |       2.58781 |           48 | film    | resnet18   | image+mean+meta | pixel+samsung      |                   0 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 9a86cbd69b2b |       2.60858 |           55 | film    | smallcnn   | image+mean+meta | samsung            |                  10 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 39ac30690cb0 |       2.61028 |           39 | film    | smallcnn   | image+mean+meta | pixel+samsung      |                   0 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 547c156fe365 |       2.66423 |           42 | late    | smallcnn   | image+mean+meta | pixel+samsung      |                  25 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| a23566bbeab9 |       2.66518 |           91 | late    | smallcnn   | image+meta      | samsung            |                  25 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 2a04fd5d01c4 |       2.67806 |           20 | late    | resnet18   | image+mean+meta | pixel+samsung      |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 5ec80f46f4c6 |       2.73203 |           84 | late    | smallcnn   | image+mean+meta | samsung            |                  25 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 68827d97d9f7 |       2.74232 |           53 | film    | smallcnn   | image+mean+meta | samsung            |                   0 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 45ef00fcd184 |       2.75131 |           23 | late    | resnet18   | image+mean+meta | pixel+samsung      |                  25 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 1b823f3f09ef |       2.76074 |           97 | film    | smallcnn   | image+meta      |                    |                  10 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 4a16823d07f8 |       2.7777  |           83 | film    | resnet18   | image+meta      |                    |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| c9f8d0357599 |       2.78357 |           97 | film    | smallcnn   | image+meta      |                    |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| e7abdb304b9f |       2.81019 |           34 | film    | smallcnn   | image+meta      | pixel+samsung      |                  10 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 038b99a7eba8 |       2.81781 |           21 | film    | resnet18   | image+mean+meta | pixel+samsung      |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 8ebdf9d18602 |       2.82954 |           35 | film    | resnet18   | image+meta      | pixel+samsung      |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |

## Grouped by model
| model   |   count |   median |    mean |      std |
|:--------|--------:|---------:|--------:|---------:|
| film    |      38 |  2.87    | 2.88725 | 0.349174 |
| late    |      24 |  2.98364 | 2.94186 | 0.298957 |

## Grouped by backbone
| backbone   |   count |   median |    mean |      std |
|:-----------|--------:|---------:|--------:|---------:|
| smallcnn   |      32 |  2.77215 | 2.81502 | 0.365817 |
| resnet18   |      30 |  2.98369 | 3.00799 | 0.254988 |

## Grouped by features
| features        |   count |   median |    mean |      std |
|:----------------|--------:|---------:|--------:|---------:|
| image+meta      |      32 |  2.88902 | 2.92998 | 0.331172 |
| image+mean+meta |      30 |  2.93579 | 2.88537 | 0.331136 |

## Grouped by mse_weight_epochs
|   mse_weight_epochs |   count |   median |    mean |      std |
|--------------------:|--------:|---------:|--------:|---------:|
|                  10 |      12 |  2.84011 | 2.82485 | 0.349526 |
|                  75 |      26 |  2.87827 | 2.90102 | 0.327754 |
|                  25 |      12 |  2.94233 | 2.91759 | 0.317292 |
|                   0 |      12 |  3.16163 | 2.99872 | 0.342411 |

## Grouped by excluded_folders
| excluded_folders   |   count |   median |    mean |      std |
|:-------------------|--------:|---------:|--------:|---------:|
| pixel+samsung      |      21 |  2.66423 | 2.62331 | 0.259356 |
| samsung            |      20 |  3.00279 | 2.95686 | 0.253942 |
| nan                |      21 |  3.17472 | 3.14732 | 0.233196 |

### Elimination candidates (excluded_folders)
|   excluded_folders |   count |   median |    mean |      std |   delta_vs_best |
|-------------------:|--------:|---------:|--------:|---------:|----------------:|
|                nan |      21 |  3.17472 | 3.14732 | 0.233196 |        0.510494 |

## Grouped by model+backbone
| model   | backbone   |   count |   median |    mean |      std |
|:--------|:-----------|--------:|---------:|--------:|---------:|
| film    | smallcnn   |      20 |  2.77215 | 2.81298 | 0.419517 |
| late    | smallcnn   |      12 |  2.85542 | 2.81841 | 0.270414 |
| film    | resnet18   |      18 |  2.93102 | 2.96978 | 0.234435 |
| late    | resnet18   |      12 |  3.01368 | 3.06531 | 0.28373  |

## Grouped by model+features
| model   | features        |   count |   median |    mean |      std |
|:--------|:----------------|--------:|---------:|--------:|---------:|
| film    | image+mean+meta |      18 |  2.85774 | 2.87203 | 0.394434 |
| film    | image+meta      |      20 |  2.87    | 2.90096 | 0.312744 |
| late    | image+meta      |      12 |  2.93751 | 2.97835 | 0.36886  |
| late    | image+mean+meta |      12 |  2.99942 | 2.90538 | 0.218893 |

## Grouped by model+mse_weight_epochs
| model   |   mse_weight_epochs |   count |   median |    mean |      std |
|:--------|--------------------:|--------:|---------:|--------:|---------:|
| film    |                  75 |      14 |  2.83095 | 2.84521 | 0.357017 |
| film    |                  10 |      12 |  2.84011 | 2.82485 | 0.349526 |
| late    |                  25 |      12 |  2.94233 | 2.91759 | 0.317292 |
| late    |                  75 |      12 |  2.99797 | 2.96614 | 0.291403 |
| film    |                   0 |      12 |  3.16163 | 2.99872 | 0.342411 |

## Grouped by model+excluded_folders
| model   | excluded_folders   |   count |   median |    mean |      std |
|:--------|:-------------------|--------:|---------:|--------:|---------:|
| film    | pixel+samsung      |      13 |  2.61028 | 2.59881 | 0.300675 |
| late    | pixel+samsung      |       8 |  2.67114 | 2.66313 | 0.18515  |
| late    | samsung            |       8 |  2.98364 | 2.93273 | 0.166287 |
| film    | samsung            |      12 |  3.1096  | 2.97295 | 0.305097 |
| film    | nan                |      13 |  3.156   | 3.0966  | 0.23503  |
| late    | nan                |       8 |  3.19539 | 3.22973 | 0.2194   |

## Grouped by backbone+features
| backbone   | features        |   count |   median |    mean |      std |
|:-----------|:----------------|--------:|---------:|--------:|---------:|
| smallcnn   | image+mean+meta |      15 |  2.74232 | 2.82272 | 0.397399 |
| smallcnn   | image+meta      |      17 |  2.78357 | 2.80822 | 0.347836 |
| resnet18   | image+meta      |      15 |  2.97892 | 3.06797 | 0.257244 |
| resnet18   | image+mean+meta |      15 |  2.98846 | 2.94801 | 0.246576 |

## Grouped by backbone+mse_weight_epochs
| backbone   |   mse_weight_epochs |   count |   median |    mean |      std |
|:-----------|--------------------:|--------:|---------:|--------:|---------:|
| smallcnn   |                  10 |       6 |  2.68466 | 2.70547 | 0.413951 |
| smallcnn   |                  25 |       6 |  2.6986  | 2.74331 | 0.258165 |
| resnet18   |                  75 |      12 |  2.87232 | 2.95255 | 0.235686 |
| smallcnn   |                   0 |       6 |  2.94541 | 2.89866 | 0.393387 |
| resnet18   |                  10 |       6 |  2.95893 | 2.94422 | 0.251425 |
| smallcnn   |                  75 |      14 |  2.99797 | 2.85685 | 0.393736 |
| resnet18   |                  25 |       6 |  3.07753 | 3.09186 | 0.286261 |
| resnet18   |                   0 |       6 |  3.18522 | 3.09878 | 0.281352 |

## Grouped by backbone+excluded_folders
| backbone   | excluded_folders   |   count |   median |    mean |      std |
|:-----------|:-------------------|--------:|---------:|--------:|---------:|
| smallcnn   | pixel+samsung      |      11 |  2.53988 | 2.44681 | 0.224828 |
| resnet18   | pixel+samsung      |      10 |  2.84979 | 2.81747 | 0.115559 |
| smallcnn   | samsung            |      10 |  2.86057 | 2.89169 | 0.272817 |
| resnet18   | samsung            |      10 |  3.0548  | 3.02203 | 0.228631 |
| smallcnn   | nan                |      11 |  3.14926 | 3.11353 | 0.210423 |
| resnet18   | nan                |      10 |  3.1852  | 3.18448 | 0.26214  |

## Grouped by features+mse_weight_epochs
| features        |   mse_weight_epochs |   count |   median |    mean |      std |
|:----------------|--------------------:|--------:|---------:|--------:|---------:|
| image+mean+meta |                  10 |       6 |  2.74585 | 2.78278 | 0.456746 |
| image+meta      |                  10 |       6 |  2.84011 | 2.86691 | 0.236444 |
| image+mean+meta |                  25 |       6 |  2.86988 | 2.88685 | 0.200178 |
| image+meta      |                  75 |      14 |  2.87232 | 2.90662 | 0.339861 |
| image+mean+meta |                   0 |       6 |  2.969   | 2.96823 | 0.360012 |
| image+meta      |                  25 |       6 |  2.96997 | 2.94832 | 0.423256 |
| image+mean+meta |                  75 |      12 |  3.02802 | 2.89449 | 0.327948 |
| image+meta      |                   0 |       6 |  3.16163 | 3.0292  | 0.355106 |

## Grouped by features+excluded_folders
| features        | excluded_folders   |   count |   median |    mean |      std |
|:----------------|:-------------------|--------:|---------:|--------:|---------:|
| image+mean+meta | pixel+samsung      |      10 |  2.63725 | 2.57748 | 0.263934 |
| image+meta      | pixel+samsung      |      11 |  2.81019 | 2.66497 | 0.260435 |
| image+mean+meta | samsung            |      10 |  3.00279 | 2.92531 | 0.264245 |
| image+meta      | samsung            |      10 |  3.06366 | 2.98841 | 0.253182 |
| image+mean+meta | nan                |      10 |  3.16536 | 3.15332 | 0.164198 |
| image+meta      | nan                |      11 |  3.18878 | 3.14186 | 0.290563 |

## Grouped by mse_weight_epochs+excluded_folders
|   mse_weight_epochs | excluded_folders   |   count |   median |    mean |      std |
|--------------------:|:-------------------|--------:|---------:|--------:|---------:|
|                  75 | pixel+samsung      |       9 |  2.54917 | 2.57901 | 0.262861 |
|                  10 | samsung            |       4 |  2.56368 | 2.71219 | 0.332713 |
|                   0 | pixel+samsung      |       4 |  2.59904 | 2.63015 | 0.262011 |
|                  25 | pixel+samsung      |       4 |  2.70777 | 2.66402 | 0.23361  |
|                  10 | pixel+samsung      |       4 |  2.84011 | 2.67545 | 0.359409 |
|                  25 | samsung            |       4 |  2.86024 | 2.88807 | 0.232156 |
|                  75 | samsung            |       8 |  3.02802 | 3.04447 | 0.147997 |
|                  10 | nan                |       4 |  3.09538 | 3.0869  | 0.264255 |
|                  25 | nan                |       4 |  3.10923 | 3.20067 | 0.258703 |
|                  75 | nan                |       9 |  3.14926 | 3.09552 | 0.260083 |
|                   0 | samsung            |       4 |  3.16163 | 3.09509 | 0.246248 |
|                   0 | nan                |       4 |  3.26818 | 3.27091 | 0.091186 |
