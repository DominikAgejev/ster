# Sweep Summary (monitor: `de00`, mode: `min`)

Total unique runs: **79**
Best run metric: **2.9537**

## Top Runs
| run_id       |   best_metric |   best_epoch | model   | backbone        | color_space   | features        |   group_split |   epochs |   batch_size |    lr |   weight_decay | mse_space   |   mse_weight_start |   mse_weight_epochs | pred_activation   |   activation_eps |   seed |
|:-------------|--------------:|-------------:|:--------|:----------------|:--------------|:----------------|--------------:|---------:|-------------:|------:|---------------:|:------------|-------------------:|--------------------:|:------------------|-----------------:|-------:|
| 088798a42de7 |       2.95373 |           59 | film    | resnet18        | rgb           | image+meta      |             0 |      100 |           32 | 0.001 |         0.0001 | rgb         |                  1 |                  25 | sigmoid_eps       |            0.001 |    100 |
| 43e02fe32b48 |       3.0006  |           48 | xattn   | resnet18        | rgb           | image+mean+meta |             1 |       50 |           32 | 0.001 |         0.0001 | rgb         |                  1 |                  50 | sigmoid_eps       |            0.001 |    100 |
| 59987cc2f5ad |       3.01538 |           49 | xattn   | resnet18        | rgb           | image+mean+meta |             1 |       50 |           32 | 0.001 |         0.0001 | rgb         |                  1 |                  10 | sigmoid_eps       |            0.001 |    100 |
| 6dd59402e219 |       3.0576  |           46 | xattn   | resnet18        | rgb           | image+mean+meta |             1 |       50 |           32 | 0.001 |         0.0001 | rgb         |                  1 |                  10 | sigmoid_eps       |            0.001 |    100 |
| 019f1d8782da |       3.07542 |           43 | late    | resnet18        | rgb           | image+meta      |             0 |      100 |           32 | 0.001 |         0.0001 | rgb         |                  1 |                  50 | sigmoid_eps       |            0.001 |    100 |
| 47500b169efb |       3.09484 |           62 | late    | smallcnn        | rgb           | image+mean+meta |             0 |      100 |           32 | 0.001 |         0.0001 | rgb         |                  1 |                  75 | sigmoid_eps       |            0.001 |    100 |
| b656750ceb94 |       3.14701 |           30 | film    | smallcnn        | rgb           | image+mean+meta |             0 |      100 |           32 | 0.001 |         0.0001 | rgb         |                  1 |                  75 | sigmoid_eps       |            0.001 |    100 |
| 1a44eb7390f3 |       3.15672 |           50 | late    | resnet18        | rgb           | image+mean+meta |             0 |      100 |           32 | 0.001 |         0.0001 | rgb         |                  1 |                  75 | sigmoid_eps       |            0.001 |    100 |
| d8a32ed3fd0f |       3.19788 |           34 | late    | resnet18        | rgb           | image+meta      |             0 |      100 |           32 | 0.001 |         0.0001 | rgb         |                  1 |                  75 | sigmoid_eps       |            0.001 |    100 |
| 73bc9382a7ee |       3.22785 |           30 | film    | smallcnn        | rgb           | image+mean+meta |             0 |      100 |           32 | 0.001 |         0.0001 | rgb         |                  1 |                  25 | sigmoid_eps       |            0.001 |    100 |
| e6342b078946 |       3.23783 |           43 | film    | smallcnn        | rgb           | image+meta      |             0 |      100 |           32 | 0.001 |         0.0001 | rgb         |                  1 |                  50 | sigmoid_eps       |            0.001 |    100 |
| e36da309bbc4 |       3.23881 |           26 | late    | resnet18        | rgb           | image+mean+meta |             0 |      100 |           32 | 0.001 |         0.0001 | rgb         |                  1 |                  25 | sigmoid_eps       |            0.001 |    100 |
| 5d2154118e22 |       3.24312 |           29 | late    | resnet18        | rgb           | image+mean+meta |             0 |      100 |           32 | 0.001 |         0.0001 | rgb         |                  1 |                  50 | sigmoid_eps       |            0.001 |    100 |
| 975885d95938 |       3.26321 |           43 | late    | resnet18        | rgb           | image+meta      |             0 |      100 |           32 | 0.001 |         0.0001 | rgb         |                  1 |                  25 | sigmoid_eps       |            0.001 |    100 |
| 4c81eb2a55e0 |       3.27982 |           41 | late    | smallcnn        | rgb           | image+mean+meta |             0 |      100 |           32 | 0.001 |         0.0001 | rgb         |                  1 |                  25 | sigmoid_eps       |            0.001 |    100 |
| 93d7b7257c20 |       3.35617 |           29 | film    | smallcnn        | rgb           | image+mean+meta |             0 |      100 |           32 | 0.001 |         0.0001 | rgb         |                  1 |                  50 | sigmoid_eps       |            0.001 |    100 |
| 1d244696f7a9 |       3.37668 |           45 | late    | smallcnn        | rgb           | image+meta      |             0 |      100 |           32 | 0.001 |         0.0001 | rgb         |                  1 |                  25 | sigmoid_eps       |            0.001 |    100 |
| 8c107a981da8 |       3.39731 |           40 | late    | smallcnn        | rgb           | image+mean+meta |             0 |      100 |           32 | 0.001 |         0.0001 | rgb         |                  1 |                  50 | sigmoid_eps       |            0.001 |    100 |
| 2e67286644e8 |       3.41162 |           37 | late    | smallcnn        | rgb           | image+meta      |             0 |      100 |           32 | 0.001 |         0.0001 | rgb         |                  1 |                  75 | sigmoid_eps       |            0.001 |    100 |
| 2ea7f23ae432 |       3.42459 |           26 | film    | resnet18        | rgb           | image+meta      |             0 |      100 |           32 | 0.001 |         0.0001 | rgb         |                  1 |                  75 | sigmoid_eps       |            0.001 |    100 |
| 68009ff90db9 |       3.44705 |           27 | film    | resnet18        | rgb           | image+meta      |             0 |      100 |           32 | 0.001 |         0.0001 | rgb         |                  1 |                  50 | sigmoid_eps       |            0.001 |    100 |
| bf6502d24c9f |       3.49127 |           32 | late    | smallcnn        | rgb           | image+meta      |             0 |      100 |           32 | 0.001 |         0.0001 | rgb         |                  1 |                  50 | sigmoid_eps       |            0.001 |    100 |
| 0638efa55404 |       3.54347 |           29 | film    | smallcnn        | rgb           | image+meta      |             0 |      100 |           32 | 0.001 |         0.0001 | rgb         |                  1 |                  25 | sigmoid_eps       |            0.001 |    100 |
| 3a2d406b6238 |       3.73783 |           23 | late    | efficientnet_b0 | rgb           | image+mean+meta |             0 |      100 |           32 | 0.001 |         0.0001 | rgb         |                  1 |                  75 | sigmoid_eps       |            0.001 |    100 |
| 522863a80cd9 |       3.75202 |           23 | film    | smallcnn        | rgb           | image+meta      |             0 |      100 |           32 | 0.001 |         0.0001 | rgb         |                  1 |                  75 | sigmoid_eps       |            0.001 |    100 |

## Grouped by model
| model   |   count |   median |    mean |      std |
|:--------|--------:|---------:|--------:|---------:|
| late    |      44 |  4.40566 | 4.29908 | 0.79096  |
| xattn   |      14 |  4.53906 | 4.24082 | 0.696052 |
| film    |      21 |  4.93801 | 4.38009 | 0.942799 |

### Elimination candidates (model)
| model   |   count |   median |    mean |      std |   delta_vs_best |
|:--------|--------:|---------:|--------:|---------:|----------------:|
| film    |      21 |  4.93801 | 4.38009 | 0.942799 |        0.532355 |

## Grouped by backbone
| backbone        |   count |   median |    mean |      std |
|:----------------|--------:|---------:|--------:|---------:|
| efficientnet_b0 |      12 |  4.36345 | 4.36849 | 0.443121 |
| smallcnn        |      24 |  4.41513 | 4.37447 | 1.06133  |
| resnet18        |      43 |  4.48116 | 4.25822 | 0.73726  |

## Grouped by features
| features        |   count |   median |    mean |      std |
|:----------------|--------:|---------:|--------:|---------:|
| image+mean+meta |      18 |  3.26147 | 3.49036 | 0.497704 |
| image+meta      |      23 |  3.75202 | 3.91803 | 0.684763 |
| image           |      23 |  4.86769 | 4.91963 | 0.464132 |
| image+mean      |      15 |  5.07824 | 4.96134 | 0.310396 |

### Elimination candidates (features)
| features   |   count |   median |    mean |      std |   delta_vs_best |
|:-----------|--------:|---------:|--------:|---------:|----------------:|
| image+mean |      15 |  5.07824 | 4.96134 | 0.310396 |         1.81677 |
| image      |      23 |  4.86769 | 4.91963 | 0.464132 |         1.60622 |

## Grouped by mse_weight_epochs
|   mse_weight_epochs |   count |   median |    mean |        std |
|--------------------:|--------:|---------:|--------:|-----------:|
|                  25 |      19 |  4.32792 | 4.32019 |   0.934134 |
|                  40 |       1 |  4.46004 | 4.46004 | nan        |
|                  75 |      19 |  4.48116 | 4.24128 |   0.865354 |
|                  50 |      27 |  4.48993 | 4.30157 |   0.7917   |
|                  10 |      13 |  4.58819 | 4.40328 |   0.67336  |

## Grouped by model+backbone
| model   | backbone        |   count |   median |    mean |      std |
|:--------|:----------------|--------:|---------:|--------:|---------:|
| late    | smallcnn        |      12 |  4.32324 | 4.42773 | 1.15669  |
| late    | efficientnet_b0 |      12 |  4.36345 | 4.36849 | 0.443121 |
| late    | resnet18        |      20 |  4.40566 | 4.18025 | 0.706568 |
| film    | smallcnn        |      12 |  4.41513 | 4.32122 | 1.00554  |
| xattn   | resnet18        |      14 |  4.53906 | 4.24082 | 0.696052 |
| film    | resnet18        |       9 |  4.93801 | 4.45859 | 0.905409 |

## Grouped by model+features
| model   | features        |   count |   median |    mean |      std |
|:--------|:----------------|--------:|---------:|--------:|---------:|
| film    | image+mean+meta |       3 |  3.22785 | 3.24368 | 0.105474 |
| late    | image+mean+meta |       9 |  3.27982 | 3.43957 | 0.325578 |
| film    | image+meta      |       6 |  3.43582 | 3.39311 | 0.272944 |
| xattn   | image+mean+meta |       6 |  3.57366 | 3.68991 | 0.761065 |
| late    | image+meta      |      13 |  3.99498 | 3.91857 | 0.686578 |
| xattn   | image           |       4 |  4.60897 | 4.60434 | 0.101891 |
| xattn   | image+meta      |       4 |  4.64802 | 4.70365 | 0.234263 |
| late    | image+mean      |       9 |  4.69204 | 4.84091 | 0.320747 |
| late    | image           |      13 |  4.83284 | 4.89953 | 0.556288 |
| film    | image+mean      |       6 |  5.12116 | 5.14197 | 0.199766 |
| film    | image           |       6 |  5.16044 | 5.17339 | 0.202655 |

## Grouped by model+mse_weight_epochs
| model   |   mse_weight_epochs |   count |   median |    mean |        std |
|:--------|--------------------:|--------:|---------:|--------:|-----------:|
| late    |                  75 |      12 |  4.23807 | 4.15484 |   0.865069 |
| late    |                  25 |      12 |  4.27841 | 4.2729  |   0.863865 |
| late    |                  50 |      15 |  4.39405 | 4.32068 |   0.797837 |
| late    |                  40 |       1 |  4.46004 | 4.46004 | nan        |
| xattn   |                  50 |       5 |  4.48993 | 4.17704 |   0.697811 |
| xattn   |                  10 |       9 |  4.58819 | 4.27625 |   0.734758 |
| late    |                  10 |       4 |  4.69355 | 4.6891  |   0.465478 |
| film    |                  75 |       7 |  4.85541 | 4.38944 |   0.913232 |
| film    |                  50 |       7 |  4.93801 | 4.34956 |   0.943222 |
| film    |                  25 |       7 |  5.00315 | 4.40126 |   1.11244  |

## Grouped by backbone+features
| backbone        | features        |   count |   median |    mean |       std |
|:----------------|:----------------|--------:|---------:|--------:|----------:|
| resnet18        | image+mean+meta |       9 |  3.23881 | 3.5309  | 0.647683  |
| smallcnn        | image+mean+meta |       6 |  3.25384 | 3.2505  | 0.117499  |
| smallcnn        | image+meta      |       6 |  3.45145 | 3.46881 | 0.174008  |
| efficientnet_b0 | image+mean+meta |       3 |  3.79668 | 3.84849 | 0.143742  |
| efficientnet_b0 | image+meta      |       3 |  4.11178 | 4.11188 | 0.116957  |
| resnet18        | image+meta      |      14 |  4.29003 | 4.06901 | 0.808343  |
| resnet18        | image           |      14 |  4.57684 | 4.65363 | 0.269468  |
| efficientnet_b0 | image+mean      |       3 |  4.62611 | 4.60538 | 0.0986635 |
| efficientnet_b0 | image           |       3 |  4.86769 | 4.90821 | 0.101867  |
| resnet18        | image+mean      |       6 |  4.8689  | 4.86809 | 0.265368  |
| smallcnn        | image+mean      |       6 |  5.1966  | 5.23256 | 0.143965  |
| smallcnn        | image           |       6 |  5.63377 | 5.54601 | 0.293057  |

## Grouped by backbone+mse_weight_epochs
| backbone        |   mse_weight_epochs |   count |   median |    mean |        std |
|:----------------|--------------------:|--------:|---------:|--------:|-----------:|
| efficientnet_b0 |                  75 |       4 |  4.24649 | 4.27463 |   0.505958 |
| smallcnn        |                  50 |       8 |  4.30982 | 4.36124 |   1.08274  |
| resnet18        |                  25 |       7 |  4.32792 | 4.11101 |   0.935453 |
| smallcnn        |                  25 |       8 |  4.39073 | 4.42707 |   1.1567   |
| efficientnet_b0 |                  50 |       4 |  4.40191 | 4.35834 |   0.487429 |
| smallcnn        |                  75 |       8 |  4.41513 | 4.3351  |   1.08887  |
| efficientnet_b0 |                  25 |       4 |  4.4275  | 4.47251 |   0.447302 |
| resnet18        |                  40 |       1 |  4.46004 | 4.46004 | nan        |
| resnet18        |                  75 |       7 |  4.48116 | 4.11499 |   0.839151 |
| resnet18        |                  50 |      15 |  4.48993 | 4.2546  |   0.722245 |
| resnet18        |                  10 |      13 |  4.58819 | 4.40328 |   0.67336  |

## Grouped by features+mse_weight_epochs
| features        |   mse_weight_epochs |   count |   median |    mean |        std |
|:----------------|--------------------:|--------:|---------:|--------:|-----------:|
| image+mean+meta |                  75 |       4 |  3.15187 | 3.2841  |   0.303704 |
| image+mean+meta |                  25 |       4 |  3.25932 | 3.43936 |   0.381719 |
| image+meta      |                  25 |       5 |  3.37668 | 3.4732  |   0.474148 |
| image+mean+meta |                  50 |       6 |  3.37674 | 3.4806  |   0.395005 |
| image+meta      |                  75 |       5 |  3.42459 | 3.55622 |   0.315233 |
| image+mean+meta |                  10 |       4 |  3.6443  | 3.76228 |   0.864128 |
| image+meta      |                  50 |       9 |  4.11178 | 3.99794 |   0.703781 |
| image           |                  40 |       1 |  4.46004 | 4.46004 | nan        |
| image           |                  10 |       5 |  4.62084 | 4.64147 |   0.219539 |
| image+meta      |                  10 |       4 |  4.80841 | 4.74653 |   0.450766 |
| image           |                  50 |       7 |  4.83284 | 4.89512 |   0.480983 |
| image+mean      |                  75 |       5 |  4.85541 | 4.8167  |   0.311644 |
| image           |                  25 |       5 |  5.02411 | 5.12702 |   0.554252 |
| image+mean      |                  25 |       5 |  5.10783 | 5.06503 |   0.324979 |
| image+mean      |                  50 |       5 |  5.13448 | 5.00227 |   0.304617 |
| image           |                  75 |       5 |  5.19251 | 5.11665 |   0.479101 |
