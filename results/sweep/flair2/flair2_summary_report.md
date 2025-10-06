# Sweep Summary (monitor: `de00`, mode: `min`)

Total unique runs: **6**
Best run metric: **2.0622**

## Top Runs
| run_id       |   best_metric |   best_epoch | model   | backbone   | features   | included_folders   |   mse_weight_epochs |   mse_weight_start |    lr |   weight_decay | pred_activation   |   activation_eps |   seed |
|:-------------|--------------:|-------------:|:--------|:-----------|:-----------|:-------------------|--------------------:|-------------------:|------:|---------------:|:------------------|-----------------:|-------:|
| badf76ea59da |       2.06222 |           59 | xattn   | resnet18   | image+meta | focus              |                   5 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| b7dc7a199ae3 |       2.15335 |           51 | xattn   | resnet18   | image+meta | focus              |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| e5769475ed86 |       2.20425 |           53 | late    | resnet18   | image+meta | focus              |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 715f696afca2 |       2.26112 |           73 | film    | resnet18   | image+meta | focus              |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 31ba986e2358 |       2.38497 |           40 | film    | resnet18   | image+meta | focus              |                   5 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| dafa270545f3 |       2.40741 |           40 | late    | resnet18   | image+meta | focus              |                   5 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |

## Grouped by model
| model   |   count |   median |    mean |       std |
|:--------|--------:|---------:|--------:|----------:|
| xattn   |       2 |  2.10779 | 2.10779 | 0.0644421 |
| late    |       2 |  2.30583 | 2.30583 | 0.143658  |
| film    |       2 |  2.32304 | 2.32304 | 0.0875749 |

## Grouped by backbone
| backbone   |   count |   median |    mean |      std |
|:-----------|--------:|---------:|--------:|---------:|
| resnet18   |       6 |  2.23268 | 2.24555 | 0.133936 |

## Grouped by features
| features   |   count |   median |    mean |      std |
|:-----------|--------:|---------:|--------:|---------:|
| image+meta |       6 |  2.23268 | 2.24555 | 0.133936 |

## Grouped by mse_weight_epochs
|   mse_weight_epochs |   count |   median |    mean |       std |
|--------------------:|--------:|---------:|--------:|----------:|
|                  75 |       3 |  2.20425 | 2.20624 | 0.0539093 |
|                   5 |       3 |  2.38497 | 2.28487 | 0.193143  |

## Grouped by included_folders
| included_folders   |   count |   median |    mean |      std |
|:-------------------|--------:|---------:|--------:|---------:|
| focus              |       6 |  2.23268 | 2.24555 | 0.133936 |

## Grouped by model+backbone
| model   | backbone   |   count |   median |    mean |       std |
|:--------|:-----------|--------:|---------:|--------:|----------:|
| xattn   | resnet18   |       2 |  2.10779 | 2.10779 | 0.0644421 |
| late    | resnet18   |       2 |  2.30583 | 2.30583 | 0.143658  |
| film    | resnet18   |       2 |  2.32304 | 2.32304 | 0.0875749 |

## Grouped by model+features
| model   | features   |   count |   median |    mean |       std |
|:--------|:-----------|--------:|---------:|--------:|----------:|
| xattn   | image+meta |       2 |  2.10779 | 2.10779 | 0.0644421 |
| late    | image+meta |       2 |  2.30583 | 2.30583 | 0.143658  |
| film    | image+meta |       2 |  2.32304 | 2.32304 | 0.0875749 |

## Grouped by model+mse_weight_epochs
| model   |   mse_weight_epochs |   count |   median |    mean |   std |
|:--------|--------------------:|--------:|---------:|--------:|------:|
| xattn   |                   5 |       1 |  2.06222 | 2.06222 |   nan |
| xattn   |                  75 |       1 |  2.15335 | 2.15335 |   nan |
| late    |                  75 |       1 |  2.20425 | 2.20425 |   nan |
| film    |                  75 |       1 |  2.26112 | 2.26112 |   nan |
| film    |                   5 |       1 |  2.38497 | 2.38497 |   nan |
| late    |                   5 |       1 |  2.40741 | 2.40741 |   nan |

## Grouped by model+included_folders
| model   | included_folders   |   count |   median |    mean |       std |
|:--------|:-------------------|--------:|---------:|--------:|----------:|
| xattn   | focus              |       2 |  2.10779 | 2.10779 | 0.0644421 |
| late    | focus              |       2 |  2.30583 | 2.30583 | 0.143658  |
| film    | focus              |       2 |  2.32304 | 2.32304 | 0.0875749 |

## Grouped by backbone+features
| backbone   | features   |   count |   median |    mean |      std |
|:-----------|:-----------|--------:|---------:|--------:|---------:|
| resnet18   | image+meta |       6 |  2.23268 | 2.24555 | 0.133936 |

## Grouped by backbone+mse_weight_epochs
| backbone   |   mse_weight_epochs |   count |   median |    mean |       std |
|:-----------|--------------------:|--------:|---------:|--------:|----------:|
| resnet18   |                  75 |       3 |  2.20425 | 2.20624 | 0.0539093 |
| resnet18   |                   5 |       3 |  2.38497 | 2.28487 | 0.193143  |

## Grouped by backbone+included_folders
| backbone   | included_folders   |   count |   median |    mean |      std |
|:-----------|:-------------------|--------:|---------:|--------:|---------:|
| resnet18   | focus              |       6 |  2.23268 | 2.24555 | 0.133936 |

## Grouped by features+mse_weight_epochs
| features   |   mse_weight_epochs |   count |   median |    mean |       std |
|:-----------|--------------------:|--------:|---------:|--------:|----------:|
| image+meta |                  75 |       3 |  2.20425 | 2.20624 | 0.0539093 |
| image+meta |                   5 |       3 |  2.38497 | 2.28487 | 0.193143  |

## Grouped by features+included_folders
| features   | included_folders   |   count |   median |    mean |      std |
|:-----------|:-------------------|--------:|---------:|--------:|---------:|
| image+meta | focus              |       6 |  2.23268 | 2.24555 | 0.133936 |

## Grouped by mse_weight_epochs+included_folders
|   mse_weight_epochs | included_folders   |   count |   median |    mean |       std |
|--------------------:|:-------------------|--------:|---------:|--------:|----------:|
|                  75 | focus              |       3 |  2.20425 | 2.20624 | 0.0539093 |
|                   5 | focus              |       3 |  2.38497 | 2.28487 | 0.193143  |
