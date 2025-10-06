# Sweep Summary (monitor: `de00`, mode: `min`)

Total unique runs: **9**
Best run metric: **2.4785**

## Top Runs
| run_id       |   best_metric |   best_epoch | model   | backbone   | features   | included_folders   |   mse_weight_epochs |   mse_weight_start |    lr |   weight_decay | pred_activation   |   activation_eps |   seed |
|:-------------|--------------:|-------------:|:--------|:-----------|:-----------|:-------------------|--------------------:|-------------------:|------:|---------------:|:------------------|-----------------:|-------:|
| 6e13a9f74235 |       2.47854 |           46 | film    | resnet18   | image+meta | focus              |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| e5769475ed86 |       2.52349 |           49 | late    | resnet18   | image+meta | focus              |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 2b91144190fb |       2.55271 |           42 | film    | resnet18   | image+meta | focus              |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 715f696afca2 |       2.60944 |           40 | film    | resnet18   | image+meta | focus              |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| aa3f7307905b |       2.62559 |           51 | late    | resnet18   | image+meta | focus              |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| b3736d9a9f2a |       2.72988 |           39 | late    | resnet18   | image+meta | focus              |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| b7dc7a199ae3 |       2.79134 |           49 | xattn   | resnet18   | image+meta | focus              |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| b8f4268e4fb2 |       2.88984 |           42 | xattn   | resnet18   | image+meta | focus              |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 21228c7b1a4d |       3.09014 |           47 | xattn   | resnet18   | image+meta | focus              |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |

## Grouped by model
| model   |   count |   median |    mean |       std |
|:--------|--------:|---------:|--------:|----------:|
| film    |       3 |  2.55271 | 2.54689 | 0.0656436 |
| late    |       3 |  2.62559 | 2.62632 | 0.103198  |
| xattn   |       3 |  2.88984 | 2.92377 | 0.152262  |

## Grouped by backbone
| backbone   |   count |   median |   mean |      std |
|:-----------|--------:|---------:|-------:|---------:|
| resnet18   |       9 |  2.62559 |  2.699 | 0.197835 |

## Grouped by features
| features   |   count |   median |   mean |      std |
|:-----------|--------:|---------:|-------:|---------:|
| image+meta |       9 |  2.62559 |  2.699 | 0.197835 |

## Grouped by mse_weight_epochs
|   mse_weight_epochs |   count |   median |   mean |      std |
|--------------------:|--------:|---------:|-------:|---------:|
|                  75 |       9 |  2.62559 |  2.699 | 0.197835 |

## Grouped by included_folders
| included_folders   |   count |   median |   mean |      std |
|:-------------------|--------:|---------:|-------:|---------:|
| focus              |       9 |  2.62559 |  2.699 | 0.197835 |

## Grouped by model+backbone
| model   | backbone   |   count |   median |    mean |       std |
|:--------|:-----------|--------:|---------:|--------:|----------:|
| film    | resnet18   |       3 |  2.55271 | 2.54689 | 0.0656436 |
| late    | resnet18   |       3 |  2.62559 | 2.62632 | 0.103198  |
| xattn   | resnet18   |       3 |  2.88984 | 2.92377 | 0.152262  |

## Grouped by model+features
| model   | features   |   count |   median |    mean |       std |
|:--------|:-----------|--------:|---------:|--------:|----------:|
| film    | image+meta |       3 |  2.55271 | 2.54689 | 0.0656436 |
| late    | image+meta |       3 |  2.62559 | 2.62632 | 0.103198  |
| xattn   | image+meta |       3 |  2.88984 | 2.92377 | 0.152262  |

## Grouped by model+mse_weight_epochs
| model   |   mse_weight_epochs |   count |   median |    mean |       std |
|:--------|--------------------:|--------:|---------:|--------:|----------:|
| film    |                  75 |       3 |  2.55271 | 2.54689 | 0.0656436 |
| late    |                  75 |       3 |  2.62559 | 2.62632 | 0.103198  |
| xattn   |                  75 |       3 |  2.88984 | 2.92377 | 0.152262  |

## Grouped by model+included_folders
| model   | included_folders   |   count |   median |    mean |       std |
|:--------|:-------------------|--------:|---------:|--------:|----------:|
| film    | focus              |       3 |  2.55271 | 2.54689 | 0.0656436 |
| late    | focus              |       3 |  2.62559 | 2.62632 | 0.103198  |
| xattn   | focus              |       3 |  2.88984 | 2.92377 | 0.152262  |

## Grouped by backbone+features
| backbone   | features   |   count |   median |   mean |      std |
|:-----------|:-----------|--------:|---------:|-------:|---------:|
| resnet18   | image+meta |       9 |  2.62559 |  2.699 | 0.197835 |

## Grouped by backbone+mse_weight_epochs
| backbone   |   mse_weight_epochs |   count |   median |   mean |      std |
|:-----------|--------------------:|--------:|---------:|-------:|---------:|
| resnet18   |                  75 |       9 |  2.62559 |  2.699 | 0.197835 |

## Grouped by backbone+included_folders
| backbone   | included_folders   |   count |   median |   mean |      std |
|:-----------|:-------------------|--------:|---------:|-------:|---------:|
| resnet18   | focus              |       9 |  2.62559 |  2.699 | 0.197835 |

## Grouped by features+mse_weight_epochs
| features   |   mse_weight_epochs |   count |   median |   mean |      std |
|:-----------|--------------------:|--------:|---------:|-------:|---------:|
| image+meta |                  75 |       9 |  2.62559 |  2.699 | 0.197835 |

## Grouped by features+included_folders
| features   | included_folders   |   count |   median |   mean |      std |
|:-----------|:-------------------|--------:|---------:|-------:|---------:|
| image+meta | focus              |       9 |  2.62559 |  2.699 | 0.197835 |

## Grouped by mse_weight_epochs+included_folders
|   mse_weight_epochs | included_folders   |   count |   median |   mean |      std |
|--------------------:|:-------------------|--------:|---------:|-------:|---------:|
|                  75 | focus              |       9 |  2.62559 |  2.699 | 0.197835 |
