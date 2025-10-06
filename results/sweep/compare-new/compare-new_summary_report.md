# Sweep Summary (monitor: `de00`, mode: `min`)

Total unique runs: **24**
Best run metric: **2.2529**

## Top Runs
| run_id       |   best_metric |   best_epoch | model   | backbone   | features        | excluded_folders   |   mse_weight_epochs |   mse_weight_start |    lr |   weight_decay | pred_activation   |   activation_eps |   seed |
|:-------------|--------------:|-------------:|:--------|:-----------|:----------------|:-------------------|--------------------:|-------------------:|------:|---------------:|:------------------|-----------------:|-------:|
| dc8d5d6de90b |       2.25288 |           86 | late    | smallcnn   | image+mean+meta | pixel+samsung      |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| d8e2e6d2ff96 |       2.25438 |           92 | late    | smallcnn   | image+meta      | pixel+samsung      |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 61d91036847f |       2.27843 |           92 | late    | smallcnn   | image+meta      | pixel+samsung      |                  25 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 2a04fd5d01c4 |       2.72476 |           39 | late    | resnet18   | image+mean+meta | pixel+samsung      |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| e23b984653a3 |       2.73819 |           43 | late    | resnet18   | image+meta      | pixel+samsung      |                  25 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| b2f1f9d6ba16 |       2.75412 |           73 | late    | smallcnn   | image+mean+meta | samsung            |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 45ef00fcd184 |       2.82687 |           37 | late    | resnet18   | image+mean+meta | pixel+samsung      |                  25 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 187bea222136 |       2.91171 |           47 | late    | resnet18   | image+mean+meta | samsung            |                  25 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 5ec80f46f4c6 |       2.91246 |           69 | late    | smallcnn   | image+mean+meta | samsung            |                  25 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| b6431c2454eb |       2.95932 |           45 | late    | resnet18   | image+meta      | samsung            |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 855951599a11 |       3.02224 |           30 | late    | resnet18   | image+mean+meta | samsung            |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 547c156fe365 |       3.03346 |           23 | late    | smallcnn   | image+mean+meta | pixel+samsung      |                  25 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 56ee0d2c2881 |       3.05237 |           33 | late    | resnet18   | image+meta      | pixel+samsung      |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| d2f521422ecc |       3.08818 |           57 | late    | smallcnn   | image+mean+meta |                    |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 65a122bc6a50 |       3.12758 |           46 | late    | resnet18   | image+meta      |                    |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 19217b89a4cc |       3.30954 |           54 | late    | resnet18   | image+meta      |                    |                  25 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 2aeb3013398f |       3.34849 |           27 | late    | resnet18   | image+meta      | samsung            |                  25 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| a23566bbeab9 |       3.49955 |           38 | late    | smallcnn   | image+meta      | samsung            |                  25 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| d6e9f8133248 |       3.53324 |           25 | late    | resnet18   | image+mean+meta |                    |                  25 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 54d452f2953d |       3.55091 |           30 | late    | resnet18   | image+mean+meta |                    |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 079fbc8243f9 |       3.68689 |           36 | late    | smallcnn   | image+meta      | samsung            |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 00609b8676a1 |       3.75256 |           30 | late    | smallcnn   | image+mean+meta |                    |                  25 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 0483b580033f |       3.78537 |           41 | late    | smallcnn   | image+meta      |                    |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 3eb00abaddb4 |       3.89614 |           37 | late    | smallcnn   | image+meta      |                    |                  25 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |

## Grouped by model
| model   |   count |   median |    mean |      std |
|:--------|--------:|---------:|--------:|---------:|
| late    |      24 |  3.04291 | 3.09582 | 0.475825 |

## Grouped by backbone
| backbone   |   count |   median |    mean |      std |
|:-----------|--------:|---------:|--------:|---------:|
| resnet18   |      12 |  3.0373  | 3.0921  | 0.287079 |
| smallcnn   |      12 |  3.06082 | 3.09954 | 0.625265 |

## Grouped by features
| features        |   count |   median |    mean |      std |
|:----------------|--------:|---------:|--------:|---------:|
| image+mean+meta |      12 |  2.96735 | 3.03028 | 0.414773 |
| image+meta      |      12 |  3.21856 | 3.16135 | 0.540364 |

## Grouped by mse_weight_epochs
|   mse_weight_epochs |   count |   median |    mean |      std |
|--------------------:|--------:|---------:|--------:|---------:|
|                  75 |      12 |   3.0373 | 3.02158 | 0.492289 |
|                  25 |      12 |   3.1715 | 3.17005 | 0.468005 |

## Grouped by excluded_folders
| excluded_folders   |   count |   median |    mean |      std |
|:-------------------|--------:|---------:|--------:|---------:|
| pixel+samsung      |       8 |  2.73147 | 2.64517 | 0.339262 |
| samsung            |       8 |  2.99078 | 3.13685 | 0.331892 |
| nan                |       8 |  3.54208 | 3.50544 | 0.304766 |

### Elimination candidates (excluded_folders)
|   excluded_folders |   count |   median |    mean |      std |   delta_vs_best |
|-------------------:|--------:|---------:|--------:|---------:|----------------:|
|                nan |       8 |  3.54208 | 3.50544 | 0.304766 |        0.810603 |

## Grouped by model+backbone
| model   | backbone   |   count |   median |    mean |      std |
|:--------|:-----------|--------:|---------:|--------:|---------:|
| late    | resnet18   |      12 |  3.0373  | 3.0921  | 0.287079 |
| late    | smallcnn   |      12 |  3.06082 | 3.09954 | 0.625265 |

## Grouped by model+features
| model   | features        |   count |   median |    mean |      std |
|:--------|:----------------|--------:|---------:|--------:|---------:|
| late    | image+mean+meta |      12 |  2.96735 | 3.03028 | 0.414773 |
| late    | image+meta      |      12 |  3.21856 | 3.16135 | 0.540364 |

## Grouped by model+mse_weight_epochs
| model   |   mse_weight_epochs |   count |   median |    mean |      std |
|:--------|--------------------:|--------:|---------:|--------:|---------:|
| late    |                  75 |      12 |   3.0373 | 3.02158 | 0.492289 |
| late    |                  25 |      12 |   3.1715 | 3.17005 | 0.468005 |

## Grouped by model+excluded_folders
| model   | excluded_folders   |   count |   median |    mean |      std |
|:--------|:-------------------|--------:|---------:|--------:|---------:|
| late    | pixel+samsung      |       8 |  2.73147 | 2.64517 | 0.339262 |
| late    | samsung            |       8 |  2.99078 | 3.13685 | 0.331892 |
| late    | nan                |       8 |  3.54208 | 3.50544 | 0.304766 |

## Grouped by backbone+features
| backbone   | features        |   count |   median |    mean |      std |
|:-----------|:----------------|--------:|---------:|--------:|---------:|
| resnet18   | image+mean+meta |       6 |  2.96697 | 3.09495 | 0.359936 |
| smallcnn   | image+mean+meta |       6 |  2.97296 | 2.96561 | 0.488763 |
| resnet18   | image+meta      |       6 |  3.08997 | 3.08925 | 0.22746  |
| smallcnn   | image+meta      |       6 |  3.59322 | 3.23346 | 0.760374 |

## Grouped by backbone+mse_weight_epochs
| backbone   |   mse_weight_epochs |   count |   median |    mean |      std |
|:-----------|--------------------:|--------:|---------:|--------:|---------:|
| smallcnn   |                  75 |       6 |  2.92115 | 2.9703  | 0.673157 |
| resnet18   |                  75 |       6 |  3.0373  | 3.07286 | 0.271504 |
| resnet18   |                  25 |       6 |  3.11062 | 3.11134 | 0.326663 |
| smallcnn   |                  25 |       6 |  3.2665  | 3.22877 | 0.605708 |

## Grouped by backbone+excluded_folders
| backbone   | excluded_folders   |   count |   median |    mean |      std |
|:-----------|:-------------------|--------:|---------:|--------:|---------:|
| smallcnn   | pixel+samsung      |       4 |  2.2664  | 2.45479 | 0.385958 |
| resnet18   | pixel+samsung      |       4 |  2.78253 | 2.83555 | 0.151483 |
| resnet18   | samsung            |       4 |  2.99078 | 3.06044 | 0.197299 |
| smallcnn   | samsung            |       4 |  3.20601 | 3.21326 | 0.450028 |
| resnet18   | nan                |       4 |  3.42139 | 3.38032 | 0.201143 |
| smallcnn   | nan                |       4 |  3.76896 | 3.63056 | 0.366767 |

## Grouped by features+mse_weight_epochs
| features        |   mse_weight_epochs |   count |   median |    mean |      std |
|:----------------|--------------------:|--------:|---------:|--------:|---------:|
| image+mean+meta |                  75 |       6 |  2.88818 | 2.89885 | 0.43469  |
| image+mean+meta |                  25 |       6 |  2.97296 | 3.16172 | 0.384793 |
| image+meta      |                  75 |       6 |  3.08997 | 3.14432 | 0.555029 |
| image+meta      |                  25 |       6 |  3.32901 | 3.17839 | 0.577609 |

## Grouped by features+excluded_folders
| features        | excluded_folders   |   count |   median |    mean |      std |
|:----------------|:-------------------|--------:|---------:|--------:|---------:|
| image+meta      | pixel+samsung      |       4 |  2.50831 | 2.58084 | 0.385197 |
| image+mean+meta | pixel+samsung      |       4 |  2.77582 | 2.70949 | 0.330382 |
| image+mean+meta | samsung            |       4 |  2.91209 | 2.90013 | 0.110323 |
| image+meta      | samsung            |       4 |  3.42402 | 3.37356 | 0.30891  |
| image+mean+meta | nan                |       4 |  3.54208 | 3.48122 | 0.280277 |
| image+meta      | nan                |       4 |  3.54745 | 3.52966 | 0.369604 |

## Grouped by mse_weight_epochs+excluded_folders
|   mse_weight_epochs | excluded_folders   |   count |   median |    mean |      std |
|--------------------:|:-------------------|--------:|---------:|--------:|---------:|
|                  75 | pixel+samsung      |       4 |  2.48957 | 2.5711  | 0.390218 |
|                  25 | pixel+samsung      |       4 |  2.78253 | 2.71924 | 0.318847 |
|                  75 | samsung            |       4 |  2.99078 | 3.10564 | 0.404057 |
|                  25 | samsung            |       4 |  3.13048 | 3.16805 | 0.301932 |
|                  75 | nan                |       4 |  3.33924 | 3.38801 | 0.337713 |
|                  25 | nan                |       4 |  3.6429  | 3.62287 | 0.256715 |
