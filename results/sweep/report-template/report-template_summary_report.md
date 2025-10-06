# Sweep Summary (monitor: `de00`, mode: `min`)

Total unique runs: **4**
Best run metric: **2.8590**

## Constants
|   epochs |   batch_size | color_space   | pred_activation   | mse_space   |    lr |   weight_decay |   mse_weight_epochs | metric_kind   |   mse_weight_start |   val_split |   grad_clip |   workers |   seed | group_split   |   hidden_classes_cnt |   meta_dim |   activation_eps | excluded_folders   | meta_model_name            |
|---------:|-------------:|:--------------|:------------------|:------------|------:|---------------:|--------------------:|:--------------|-------------------:|------------:|------------:|----------:|-------:|:--------------|---------------------:|-----------:|-----------------:|:-------------------|:---------------------------|
|      100 |           32 | rgb           | sigmoid_eps       | rgb         | 0.001 |         0.0001 |                  75 | val           |                  1 |        0.15 |           1 |         0 |    100 |               |                    0 |        256 |            0.001 |                    | jhu-clsp/ettin-encoder-17m |

## Grid keys (vary across runs)
`model`, `backbone`, `features`, `included_folders`, `meta_encoder`, `meta_layers`, `meta_text_template`

## Top Runs
| run_id       | metric_kind   |   epochs |   batch_size | color_space   | pred_activation   | mse_space   |    lr |   weight_decay |   mse_weight_epochs | model   | backbone   | features        | included_folders   | meta_encoder   |   meta_layers | meta_text_template   |   best_metric |   best_epoch |
|:-------------|:--------------|---------:|-------------:|:--------------|:------------------|:------------|------:|---------------:|--------------------:|:--------|:-----------|:----------------|:-------------------|:---------------|--------------:|:---------------------|--------------:|-------------:|
| 00c2199582c8 | val           |      100 |           32 | rgb           | sigmoid_eps       | rgb         | 0.001 |         0.0001 |                  75 | film    | resnet18   | image+mean+meta | focus              | flair          |            -1 | compact              |       2.85904 |           75 |
| 61ec79c3dccc | val           |      100 |           32 | rgb           | sigmoid_eps       | rgb         | 0.001 |         0.0001 |                  75 | film    | resnet18   | image+mean+meta | focus              | flair          |            -2 | compact              |       2.86839 |           88 |
| 36c91c925cc4 | val           |      100 |           32 | rgb           | sigmoid_eps       | rgb         | 0.001 |         0.0001 |                  75 | film    | resnet18   | image+mean+meta | focus              | flair          |            -1 | compact              |     nan       |            1 |
| 1899c243af81 | val           |      100 |           32 | rgb           | sigmoid_eps       | rgb         | 0.001 |         0.0001 |                  75 | film    | resnet18   | image+mean+meta | focus              | flair          |            -2 | compact              |     nan       |            1 |
