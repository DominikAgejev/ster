# Sweep Summary (monitor: `de00`, mode: `min`)

Total unique runs: **192**
Best run metric: **2.2792**

## Top Runs
| run_id       |   best_metric |   best_epoch | model   | backbone   | features        | excluded_folders   |   mse_weight_epochs |   mse_weight_start |    lr |   weight_decay | pred_activation   |   activation_eps |   seed |
|:-------------|--------------:|-------------:|:--------|:-----------|:----------------|:-------------------|--------------------:|-------------------:|------:|---------------:|:------------------|-----------------:|-------:|
| f080c2c6b2e3 |       2.27924 |           98 | xattn   | smallcnn   | image+meta      | pixel+samsung      |                  25 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| c6bb6d95e899 |       2.30361 |           93 | late    | smallcnn   | image+mean+meta | pixel+samsung      |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| d303f962a2f7 |       2.30662 |           65 | xattn   | smallcnn   | image+mean+meta | pixel+samsung      |                  10 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 21e93d89c724 |       2.32016 |           97 | late    | smallcnn   | image+mean+meta | pixel+samsung      |                  25 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| ae792477c42e |       2.34418 |           90 | xattn   | smallcnn   | image+meta      | samsung            |                  25 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| aa21f690580e |       2.35036 |           74 | xattn   | resnet18   | image+mean+meta | pixel+samsung      |                  25 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| fc9947321cbd |       2.39199 |           77 | late    | smallcnn   | image+meta      | pixel+samsung      |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| a6c98637a362 |       2.41287 |           89 | xattn   | smallcnn   | image+meta      | samsung            |                  50 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 495144157618 |       2.59242 |           43 | xattn   | smallcnn   | image+mean+meta | pixel+samsung      |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 48322372fc69 |       2.60613 |           31 | xattn   | smallcnn   | image+mean+meta | pixel+samsung      |                  50 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 2fa45dd1ad04 |       2.64301 |           38 | xattn   | smallcnn   | image+mean+meta | pixel+samsung      |                  25 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 4d9488a004d1 |       2.6556  |           39 | xattn   | resnet18   | image+mean+meta | pixel+samsung      |                  50 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 8ff51ac6b9b5 |       2.65717 |           72 | xattn   | resnet18   | image+meta      | samsung            |                  10 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 86a553d7b8a7 |       2.66722 |           84 | xattn   | smallcnn   | image+mean+meta | samsung            |                  50 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| dc85cdd8d8d0 |       2.68914 |           44 | xattn   | smallcnn   | image+meta      | pixel+samsung      |                  50 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 5cef16e71064 |       2.69141 |           50 | xattn   | smallcnn   | image+meta      | pixel+samsung      |                  10 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| b3fd24f9a72c |       2.70305 |           96 | xattn   | smallcnn   | image+meta      | pixel              |                  25 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 51de8e408d79 |       2.70578 |           84 | xattn   | smallcnn   | image+mean+meta | pixel              |                  25 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| bf771c36ae00 |       2.71309 |           94 | xattn   | smallcnn   | image+mean+meta | pixel              |                  10 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| ec30cfae7e2c |       2.74369 |           42 | xattn   | resnet18   | image+meta      | pixel+samsung      |                  50 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 5f978c427a76 |       2.7463  |           34 | late    | resnet18   | image+mean+meta | pixel+samsung      |                  50 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 6efeb6c7f456 |       2.75664 |           53 | xattn   | resnet18   | image+mean+meta | samsung            |                  25 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 426ccd634b38 |       2.76108 |           26 | xattn   | resnet18   | image+mean+meta | pixel+samsung      |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 1bc53f63fe14 |       2.76624 |           84 | late    | smallcnn   | image+meta      | pixel              |                  75 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |
| 8f9cebc057b9 |       2.76891 |           82 | late    | smallcnn   | image+meta      | samsung            |                  10 |                  1 | 0.001 |         0.0001 | sigmoid_eps       |            0.001 |    100 |

## Grouped by model
| model   |   count |   median |    mean |     std |
|:--------|--------:|---------:|--------:|--------:|
| xattn   |      96 |  3.2358  | 3.64562 | 1.07564 |
| late    |      96 |  3.27838 | 3.72683 | 1.07585 |

## Grouped by backbone
| backbone   |   count |   median |    mean |      std |
|:-----------|--------:|---------:|--------:|---------:|
| smallcnn   |      96 |  3.21934 | 3.82515 | 1.30932  |
| resnet18   |      96 |  3.27838 | 3.5473  | 0.751302 |

## Grouped by features
| features        |   count |   median |    mean |      std |
|:----------------|--------:|---------:|--------:|---------:|
| image+mean+meta |      64 |  2.93797 | 2.97055 | 0.302488 |
| image+meta      |      64 |  3.00317 | 3.07463 | 0.365131 |
| image           |      64 |  5.08265 | 5.01351 | 0.761782 |

### Elimination candidates (features)
| features   |   count |   median |    mean |      std |   delta_vs_best |
|:-----------|--------:|---------:|--------:|---------:|----------------:|
| image      |      64 |  5.08265 | 5.01351 | 0.761782 |         2.14469 |

## Grouped by mse_weight_epochs
|   mse_weight_epochs |   count |   median |    mean |     std |
|--------------------:|--------:|---------:|--------:|--------:|
|                  25 |      48 |  3.21343 | 3.66531 | 1.16079 |
|                  50 |      48 |  3.23987 | 3.70962 | 1.08898 |
|                  75 |      48 |  3.27453 | 3.65794 | 1.00176 |
|                  10 |      48 |  3.3334  | 3.71205 | 1.07031 |

## Grouped by excluded_folders
| excluded_folders   |   count |   median |    mean |      std |
|:-------------------|--------:|---------:|--------:|---------:|
| pixel+samsung      |      48 |  2.93233 | 3.33646 | 0.932019 |
| samsung            |      48 |  3.22176 | 3.59126 | 1.04452  |
| pixel              |      48 |  3.25612 | 3.781   | 1.10986  |
| none               |      48 |  3.53318 | 4.03619 | 1.1068   |

### Elimination candidates (excluded_folders)
| excluded_folders   |   count |   median |    mean |    std |   delta_vs_best |
|:-------------------|--------:|---------:|--------:|-------:|----------------:|
| none               |      48 |  3.53318 | 4.03619 | 1.1068 |        0.600858 |

## Grouped by model+backbone
| model   | backbone   |   count |   median |    mean |      std |
|:--------|:-----------|--------:|---------:|--------:|---------:|
| xattn   | smallcnn   |      48 |  2.92137 | 3.71619 | 1.28312  |
| late    | smallcnn   |      48 |  3.26809 | 3.93412 | 1.33958  |
| late    | resnet18   |      48 |  3.27838 | 3.51954 | 0.67623  |
| xattn   | resnet18   |      48 |  3.33635 | 3.57506 | 0.825868 |

## Grouped by model+features
| model   | features        |   count |   median |    mean |      std |
|:--------|:----------------|--------:|---------:|--------:|---------:|
| xattn   | image+mean+meta |      32 |  2.85293 | 2.87304 | 0.279311 |
| xattn   | image+meta      |      32 |  2.90939 | 3.07176 | 0.448097 |
| late    | image+meta      |      32 |  3.03078 | 3.0775  | 0.264828 |
| late    | image+mean+meta |      32 |  3.09359 | 3.06805 | 0.297165 |
| xattn   | image           |      32 |  4.9804  | 4.99207 | 0.662867 |
| late    | image           |      32 |  5.14187 | 5.03495 | 0.859651 |

## Grouped by model+mse_weight_epochs
| model   |   mse_weight_epochs |   count |   median |    mean |      std |
|:--------|--------------------:|--------:|---------:|--------:|---------:|
| xattn   |                  25 |      24 |  3.03078 | 3.63792 | 1.26115  |
| xattn   |                  50 |      24 |  3.19425 | 3.66908 | 1.11241  |
| late    |                  50 |      24 |  3.23987 | 3.75016 | 1.08739  |
| xattn   |                  10 |      24 |  3.25654 | 3.6511  | 1.06955  |
| late    |                  25 |      24 |  3.25966 | 3.69269 | 1.07768  |
| late    |                  75 |      24 |  3.2743  | 3.69149 | 1.11464  |
| xattn   |                  75 |      24 |  3.27453 | 3.62439 | 0.897726 |
| late    |                  10 |      24 |  3.3334  | 3.773   | 1.09052  |

## Grouped by model+excluded_folders
| model   | excluded_folders   |   count |   median |    mean |      std |
|:--------|:-------------------|--------:|---------:|--------:|---------:|
| xattn   | pixel+samsung      |      24 |  2.77477 | 3.29627 | 0.938668 |
| xattn   | samsung            |      24 |  2.9031  | 3.52755 | 1.13698  |
| late    | pixel+samsung      |      24 |  2.96069 | 3.37666 | 0.943725 |
| xattn   | pixel              |      24 |  3.15532 | 3.75305 | 1.155    |
| late    | samsung            |      24 |  3.27741 | 3.65497 | 0.963481 |
| late    | pixel              |      24 |  3.28922 | 3.80896 | 1.08695  |
| late    | none               |      24 |  3.41495 | 4.06676 | 1.23266  |
| xattn   | none               |      24 |  3.61144 | 4.00563 | 0.990909 |

## Grouped by backbone+features
| backbone   | features        |   count |   median |    mean |      std |
|:-----------|:----------------|--------:|---------:|--------:|---------:|
| smallcnn   | image+mean+meta |      32 |  2.89558 | 2.92873 | 0.332052 |
| smallcnn   | image+meta      |      32 |  2.90939 | 2.96858 | 0.375685 |
| resnet18   | image+mean+meta |      32 |  2.97243 | 3.01236 | 0.268479 |
| resnet18   | image+meta      |      32 |  3.19824 | 3.18069 | 0.326468 |
| resnet18   | image           |      32 |  4.33714 | 4.44886 | 0.52722  |
| smallcnn   | image           |      32 |  5.70633 | 5.57816 | 0.493107 |

## Grouped by backbone+mse_weight_epochs
| backbone   |   mse_weight_epochs |   count |   median |    mean |      std |
|:-----------|--------------------:|--------:|---------:|--------:|---------:|
| smallcnn   |                  10 |      24 |  3.04542 | 3.82515 | 1.32793  |
| smallcnn   |                  25 |      24 |  3.13684 | 3.81854 | 1.42372  |
| resnet18   |                  50 |      24 |  3.23987 | 3.53764 | 0.756199 |
| resnet18   |                  75 |      24 |  3.2478  | 3.54056 | 0.725978 |
| resnet18   |                  25 |      24 |  3.25012 | 3.51207 | 0.823096 |
| smallcnn   |                  50 |      24 |  3.29619 | 3.8816  | 1.33781  |
| smallcnn   |                  75 |      24 |  3.30103 | 3.77533 | 1.22266  |
| resnet18   |                  10 |      24 |  3.44464 | 3.59895 | 0.742193 |

## Grouped by backbone+excluded_folders
| backbone   | excluded_folders   |   count |   median |    mean |      std |
|:-----------|:-------------------|--------:|---------:|--------:|---------:|
| smallcnn   | pixel+samsung      |      24 |  2.91379 | 3.46036 | 1.19892  |
| resnet18   | pixel+samsung      |      24 |  2.94591 | 3.21257 | 0.552845 |
| resnet18   | samsung            |      24 |  2.9909  | 3.34355 | 0.602735 |
| smallcnn   | pixel              |      24 |  3.05501 | 3.88719 | 1.38494  |
| smallcnn   | samsung            |      24 |  3.25957 | 3.83896 | 1.31839  |
| resnet18   | pixel              |      24 |  3.30008 | 3.67481 | 0.758645 |
| smallcnn   | none               |      24 |  3.51741 | 4.11411 | 1.32555  |
| resnet18   | none               |      24 |  3.53318 | 3.95828 | 0.856462 |

## Grouped by features+mse_weight_epochs
| features        |   mse_weight_epochs |   count |   median |    mean |      std |
|:----------------|--------------------:|--------:|---------:|--------:|---------:|
| image+mean+meta |                  25 |      16 |  2.85564 | 2.92582 | 0.347795 |
| image+mean+meta |                  50 |      16 |  2.91951 | 2.92304 | 0.200423 |
| image+meta      |                  25 |      16 |  2.95304 | 2.97982 | 0.368574 |
| image+mean+meta |                  10 |      16 |  2.96509 | 3.02974 | 0.342634 |
| image+meta      |                  75 |      16 |  2.96609 | 3.08478 | 0.400964 |
| image+meta      |                  10 |      16 |  2.9762  | 3.07243 | 0.29896  |
| image+mean+meta |                  75 |      16 |  3.0244  | 3.00358 | 0.310361 |
| image+meta      |                  50 |      16 |  3.19824 | 3.16149 | 0.395219 |
| image           |                  10 |      16 |  4.85392 | 5.03398 | 0.765295 |
| image           |                  75 |      16 |  4.99923 | 4.88546 | 0.687677 |
| image           |                  50 |      16 |  5.07495 | 5.04432 | 0.799212 |
| image           |                  25 |      16 |  5.21015 | 5.09027 | 0.845432 |

## Grouped by features+excluded_folders
| features        | excluded_folders   |   count |   median |    mean |      std |
|:----------------|:-------------------|--------:|---------:|--------:|---------:|
| image+mean+meta | pixel+samsung      |      16 |  2.70095 | 2.65288 | 0.224222 |
| image+meta      | samsung            |      16 |  2.87925 | 2.9314  | 0.325397 |
| image+meta      | pixel+samsung      |      16 |  2.92244 | 2.86319 | 0.291957 |
| image+mean+meta | samsung            |      16 |  2.93797 | 2.97995 | 0.199202 |
| image+mean+meta | pixel              |      16 |  2.96693 | 3.02406 | 0.241812 |
| image+meta      | pixel              |      16 |  3.07737 | 3.09452 | 0.286989 |
| image+mean+meta | none               |      16 |  3.20266 | 3.22529 | 0.236342 |
| image+meta      | none               |      16 |  3.47688 | 3.40942 | 0.311308 |
| image           | pixel+samsung      |      16 |  4.47817 | 4.49332 | 0.649149 |
| image           | samsung            |      16 |  4.49497 | 4.86242 | 0.828473 |
| image           | none               |      16 |  5.26622 | 5.47387 | 0.600398 |
| image           | pixel              |      16 |  5.31844 | 5.22443 | 0.618521 |

## Grouped by mse_weight_epochs+excluded_folders
|   mse_weight_epochs | excluded_folders   |   count |   median |    mean |      std |
|--------------------:|:-------------------|--------:|---------:|--------:|---------:|
|                  75 | pixel+samsung      |      12 |  2.85349 | 3.30131 | 0.981489 |
|                  25 | pixel+samsung      |      12 |  2.87538 | 3.25534 | 1.04187  |
|                  25 | samsung            |      12 |  2.9127  | 3.55839 | 1.16202  |
|                  50 | pixel+samsung      |      12 |  2.92729 | 3.36518 | 0.915783 |
|                  10 | pixel+samsung      |      12 |  2.99098 | 3.42402 | 0.897927 |
|                  10 | samsung            |      12 |  3.05866 | 3.61129 | 1.13032  |
|                  10 | pixel              |      12 |  3.18711 | 3.78844 | 1.1994   |
|                  50 | pixel              |      12 |  3.23307 | 3.78813 | 1.14725  |
|                  50 | samsung            |      12 |  3.2392  | 3.63114 | 1.11641  |
|                  25 | pixel              |      12 |  3.25612 | 3.80748 | 1.17683  |
|                  75 | pixel              |      12 |  3.2743  | 3.73996 | 1.05864  |
|                  75 | samsung            |      12 |  3.27453 | 3.56421 | 0.885    |
|                  25 | none               |      12 |  3.44841 | 4.04001 | 1.25118  |
|                  75 | none               |      12 |  3.51776 | 4.02629 | 1.05681  |
|                  10 | none               |      12 |  3.53318 | 4.02444 | 1.07602  |
|                  50 | none               |      12 |  3.59005 | 4.05403 | 1.18042  |
