# Sweep Summary (monitor: `de00`, mode: `min`)

Total unique runs: **12**
Best run metric: **2.1059**

## Top Runs
| run_id       |   best_metric |   best_epoch | model   | backbone   | features   | included_folders   |   mse_weight_epochs |   mse_weight_start |    lr |   weight_decay | pred_activation   |   activation_eps |   seed |
|:-------------|--------------:|-------------:|:--------|:-----------|:-----------|:-------------------|--------------------:|-------------------:|------:|---------------:|:------------------|-----------------:|-------:|
| f1f0520188fa |       2.10588 |           97 | late    | smallcnn   | image+meta | focus              |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 04eb47257e95 |       2.13211 |           99 | xattn   | smallcnn   | image+meta | focus              |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 715f696afca2 |       2.21113 |           63 | film    | resnet18   | image+meta | focus              |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| b7dc7a199ae3 |       2.46308 |           49 | xattn   | resnet18   | image+meta | focus              |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| dafa270545f3 |       2.69452 |           29 | late    | resnet18   | image+meta | focus              |                   5 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| f92448cc858b |       2.70104 |           39 | late    | smallcnn   | image+meta | focus              |                   5 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| badf76ea59da |       2.77025 |           32 | xattn   | resnet18   | image+meta | focus              |                   5 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| e5769475ed86 |       2.80492 |           19 | late    | resnet18   | image+meta | focus              |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 79e7d2f86bc3 |       2.83307 |           47 | film    | smallcnn   | image+meta | focus              |                   5 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 31ba986e2358 |       2.85123 |           33 | film    | resnet18   | image+meta | focus              |                   5 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 67770745f825 |       2.8779  |           40 | film    | smallcnn   | image+meta | focus              |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| e87d77f936e6 |       2.9951  |           35 | xattn   | smallcnn   | image+meta | focus              |                   5 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |

## Grouped by model
| model   |   count |   median |    mean |      std |
|:--------|--------:|---------:|--------:|---------:|
| xattn   |       4 |  2.61667 | 2.59013 | 0.375221 |
| late    |       4 |  2.69778 | 2.57659 | 0.317856 |
| film    |       4 |  2.84215 | 2.69333 | 0.321996 |

## Grouped by backbone
| backbone   |   count |   median |    mean |      std |
|:-----------|--------:|---------:|--------:|---------:|
| resnet18   |       6 |  2.73239 | 2.63252 | 0.247649 |
| smallcnn   |       6 |  2.76705 | 2.60752 | 0.390027 |

## Grouped by features
| features   |   count |   median |    mean |      std |
|:-----------|--------:|---------:|--------:|---------:|
| image+meta |      12 |  2.73565 | 2.62002 | 0.311759 |

## Grouped by mse_weight_epochs
|   mse_weight_epochs |   count |   median |    mean |      std |
|--------------------:|--------:|---------:|--------:|---------:|
|                  75 |       6 |  2.3371  | 2.4325  | 0.341741 |
|                   5 |       6 |  2.80166 | 2.80754 | 0.112469 |

## Grouped by included_folders
| included_folders   |   count |   median |    mean |      std |
|:-------------------|--------:|---------:|--------:|---------:|
| focus              |      12 |  2.73565 | 2.62002 | 0.311759 |

## Grouped by model+backbone
| model   | backbone   |   count |   median |    mean |       std |
|:--------|:-----------|--------:|---------:|--------:|----------:|
| late    | smallcnn   |       2 |  2.40346 | 2.40346 | 0.42084   |
| film    | resnet18   |       2 |  2.53118 | 2.53118 | 0.452621  |
| xattn   | smallcnn   |       2 |  2.5636  | 2.5636  | 0.610228  |
| xattn   | resnet18   |       2 |  2.61667 | 2.61667 | 0.217206  |
| late    | resnet18   |       2 |  2.74972 | 2.74972 | 0.0780648 |
| film    | smallcnn   |       2 |  2.85549 | 2.85549 | 0.0316982 |

## Grouped by model+features
| model   | features   |   count |   median |    mean |      std |
|:--------|:-----------|--------:|---------:|--------:|---------:|
| xattn   | image+meta |       4 |  2.61667 | 2.59013 | 0.375221 |
| late    | image+meta |       4 |  2.69778 | 2.57659 | 0.317856 |
| film    | image+meta |       4 |  2.84215 | 2.69333 | 0.321996 |

## Grouped by model+mse_weight_epochs
| model   |   mse_weight_epochs |   count |   median |    mean |        std |
|:--------|--------------------:|--------:|---------:|--------:|-----------:|
| xattn   |                  75 |       2 |  2.29759 | 2.29759 | 0.234032   |
| late    |                  75 |       2 |  2.4554  | 2.4554  | 0.494295   |
| film    |                  75 |       2 |  2.54452 | 2.54452 | 0.471477   |
| late    |                   5 |       2 |  2.69778 | 2.69778 | 0.00461005 |
| film    |                   5 |       2 |  2.84215 | 2.84215 | 0.0128414  |
| xattn   |                   5 |       2 |  2.88268 | 2.88268 | 0.158989   |

## Grouped by model+included_folders
| model   | included_folders   |   count |   median |    mean |      std |
|:--------|:-------------------|--------:|---------:|--------:|---------:|
| xattn   | focus              |       4 |  2.61667 | 2.59013 | 0.375221 |
| late    | focus              |       4 |  2.69778 | 2.57659 | 0.317856 |
| film    | focus              |       4 |  2.84215 | 2.69333 | 0.321996 |

## Grouped by backbone+features
| backbone   | features   |   count |   median |    mean |      std |
|:-----------|:-----------|--------:|---------:|--------:|---------:|
| resnet18   | image+meta |       6 |  2.73239 | 2.63252 | 0.247649 |
| smallcnn   | image+meta |       6 |  2.76705 | 2.60752 | 0.390027 |

## Grouped by backbone+mse_weight_epochs
| backbone   |   mse_weight_epochs |   count |   median |    mean |       std |
|:-----------|--------------------:|--------:|---------:|--------:|----------:|
| smallcnn   |                  75 |       3 |  2.13211 | 2.37196 | 0.438352  |
| resnet18   |                  75 |       3 |  2.46308 | 2.49304 | 0.298026  |
| resnet18   |                   5 |       3 |  2.77025 | 2.772   | 0.0783722 |
| smallcnn   |                   5 |       3 |  2.83307 | 2.84307 | 0.147286  |

## Grouped by backbone+included_folders
| backbone   | included_folders   |   count |   median |    mean |      std |
|:-----------|:-------------------|--------:|---------:|--------:|---------:|
| resnet18   | focus              |       6 |  2.73239 | 2.63252 | 0.247649 |
| smallcnn   | focus              |       6 |  2.76705 | 2.60752 | 0.390027 |

## Grouped by features+mse_weight_epochs
| features   |   mse_weight_epochs |   count |   median |    mean |      std |
|:-----------|--------------------:|--------:|---------:|--------:|---------:|
| image+meta |                  75 |       6 |  2.3371  | 2.4325  | 0.341741 |
| image+meta |                   5 |       6 |  2.80166 | 2.80754 | 0.112469 |

## Grouped by features+included_folders
| features   | included_folders   |   count |   median |    mean |      std |
|:-----------|:-------------------|--------:|---------:|--------:|---------:|
| image+meta | focus              |      12 |  2.73565 | 2.62002 | 0.311759 |

## Grouped by mse_weight_epochs+included_folders
|   mse_weight_epochs | included_folders   |   count |   median |    mean |      std |
|--------------------:|:-------------------|--------:|---------:|--------:|---------:|
|                  75 | focus              |       6 |  2.3371  | 2.4325  | 0.341741 |
|                   5 | focus              |       6 |  2.80166 | 2.80754 | 0.112469 |
