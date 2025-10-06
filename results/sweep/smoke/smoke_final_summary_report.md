# Sweep Summary (monitor: `de00`, mode: `min`)

Total unique runs: **2**
Best run metric (test): **6.6704**

## Constants
| model   | backbone   | features        |   epochs |   batch_size |    lr |   weight_decay |   mse_weight_epochs |   val_split | group_split   | color_space   | excluded_folders   | included_folders   | meta_encoder   | meta_model_name            |   meta_layers | meta_text_template   |
|:--------|:-----------|:----------------|---------:|-------------:|------:|---------------:|--------------------:|------------:|:--------------|:--------------|:-------------------|:-------------------|:---------------|:---------------------------|--------------:|:---------------------|
| late    | resnet18   | image+mean+meta |       20 |           32 | 0.001 |         0.0001 |                  75 |           0 |               | rgb           |                    | focus/iphone       | flair          | jhu-clsp/ettin-encoder-17m |            -2 | compact              |

## Grid keys (vary across runs)
`metric_kind`, `mse_weight_start`, `mse_space`, `grad_clip`, `workers`, `seed`, `hidden_classes_cnt`, `meta_dim`, `pred_activation`, `activation_eps`, `lr_auto`, `lr_schedule`, `optim`

## Top Runs
| run_id       | metric_kind   | metric_kind   |   mse_weight_start | mse_space   |   grad_clip |   workers |   seed |   hidden_classes_cnt |   meta_dim | pred_activation   |   activation_eps |   lr_auto | lr_schedule   | optim   |   best_metric |   best_epoch |
|:-------------|:--------------|:--------------|-------------------:|:------------|------------:|----------:|-------:|---------------------:|-----------:|:------------------|-----------------:|----------:|:--------------|:--------|--------------:|-------------:|
| 6111e9dc0d75 | test          | test          |                nan | nan         |         nan |       nan |    nan |                  nan |        nan | nan               |          nan     |         1 | paper_resnet  | sgd     |       6.67044 |           20 |
| 67d41b8358c5 | val           | val           |                  1 | rgb         |           1 |         0 |    100 |                    0 |        256 | sigmoid_eps       |            0.001 |       nan | nan           | nan     |     nan       |           20 |
