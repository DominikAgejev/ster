# Sweep Summary (monitor: `de00`, mode: `min`)

Total unique runs: **7**
Best run metric: **2.3686**

## Top Runs
| run_id       |   best_metric |   best_epoch | model   | backbone   | color_space   | features        | group_split   |   epochs |   batch_size |    lr |   weight_decay | mse_space   |   mse_weight_start |   mse_weight_epochs | pred_activation   |   activation_eps |   seed |
|:-------------|--------------:|-------------:|:--------|:-----------|:--------------|:----------------|:--------------|---------:|-------------:|------:|---------------:|:------------|-------------------:|--------------------:|:------------------|-----------------:|-------:|
| 8ac548e29eb9 |       2.36858 |           49 | xattn   | resnet18   | rgb           | image+mean+meta | class         |       50 |           32 | 0.001 |         0.0001 | rgb         |                  1 |                  10 | sigmoid_eps       |            0.001 |    100 |
| 12f41ff69cca |       2.37592 |           41 | xattn   | resnet18   | rgb           | image+mean+meta | class         |       50 |           32 | 0.001 |         0.0001 | rgb         |                  1 |                  50 | sigmoid_eps       |            0.001 |    100 |
| 6495959e9a58 |       2.39215 |           41 | xattn   | resnet18   | rgb           | image+mean+meta | class         |       50 |           32 | 0.001 |         0.0001 | rgb         |                  1 |                  50 | sigmoid_eps       |            0.001 |    100 |
| 70aedc67000c |       2.95291 |           44 | xattn   | resnet18   | rgb           | image+mean+meta | class         |       50 |           32 | 0.001 |         0.0001 | rgb         |                  1 |                  10 | sigmoid_eps       |            0.001 |    100 |
| f812b594346c |       2.96495 |           44 | xattn   | resnet18   | rgb           | image+mean+meta | class         |       50 |           32 | 0.001 |         0.0001 | rgb         |                  1 |                  50 | sigmoid_eps       |            0.001 |    100 |
| 1590a8b16249 |       8.3828  |            3 | xattn   | resnet18   | rgb           | image+mean+meta | folder        |       50 |           32 | 0.001 |         0.0001 | rgb         |                  1 |                  10 | sigmoid_eps       |            0.001 |    100 |
| 505c526ae1bf |       9.34914 |           27 | xattn   | resnet18   | rgb           | image+mean+meta | folder        |       50 |           32 | 0.001 |         0.0001 | rgb         |                  1 |                  50 | sigmoid_eps       |            0.001 |    100 |

## Grouped by model
| model   |   count |   median |    mean |     std |
|:--------|--------:|---------:|--------:|--------:|
| xattn   |       7 |  2.95291 | 4.39806 | 3.07585 |

## Grouped by backbone
| backbone   |   count |   median |    mean |     std |
|:-----------|--------:|---------:|--------:|--------:|
| resnet18   |       7 |  2.95291 | 4.39806 | 3.07585 |

## Grouped by features
| features        |   count |   median |    mean |     std |
|:----------------|--------:|---------:|--------:|--------:|
| image+mean+meta |       7 |  2.95291 | 4.39806 | 3.07585 |

## Grouped by mse_weight_epochs
|   mse_weight_epochs |   count |   median |    mean |     std |
|--------------------:|--------:|---------:|--------:|--------:|
|                  50 |       4 |  2.67855 | 4.27054 | 3.3968  |
|                  10 |       3 |  2.95291 | 4.5681  | 3.31653 |

## Grouped by model+backbone
| model   | backbone   |   count |   median |    mean |     std |
|:--------|:-----------|--------:|---------:|--------:|--------:|
| xattn   | resnet18   |       7 |  2.95291 | 4.39806 | 3.07585 |

## Grouped by model+features
| model   | features        |   count |   median |    mean |     std |
|:--------|:----------------|--------:|---------:|--------:|--------:|
| xattn   | image+mean+meta |       7 |  2.95291 | 4.39806 | 3.07585 |

## Grouped by model+mse_weight_epochs
| model   |   mse_weight_epochs |   count |   median |    mean |     std |
|:--------|--------------------:|--------:|---------:|--------:|--------:|
| xattn   |                  50 |       4 |  2.67855 | 4.27054 | 3.3968  |
| xattn   |                  10 |       3 |  2.95291 | 4.5681  | 3.31653 |

## Grouped by backbone+features
| backbone   | features        |   count |   median |    mean |     std |
|:-----------|:----------------|--------:|---------:|--------:|--------:|
| resnet18   | image+mean+meta |       7 |  2.95291 | 4.39806 | 3.07585 |

## Grouped by backbone+mse_weight_epochs
| backbone   |   mse_weight_epochs |   count |   median |    mean |     std |
|:-----------|--------------------:|--------:|---------:|--------:|--------:|
| resnet18   |                  50 |       4 |  2.67855 | 4.27054 | 3.3968  |
| resnet18   |                  10 |       3 |  2.95291 | 4.5681  | 3.31653 |

## Grouped by features+mse_weight_epochs
| features        |   mse_weight_epochs |   count |   median |    mean |     std |
|:----------------|--------------------:|--------:|---------:|--------:|--------:|
| image+mean+meta |                  50 |       4 |  2.67855 | 4.27054 | 3.3968  |
| image+mean+meta |                  10 |       3 |  2.95291 | 4.5681  | 3.31653 |
