# Sweep Summary (monitor: `de00`, mode: `min`)

Total unique runs: **2**
Best run metric (test): **4.9181**

## Constants
|   epochs |   batch_size |   lr |   weight_decay |   mse_weight_epochs |   val_split | color_space   | meta_model_name            |   lr_auto |   lr_schedule |   optim |
|---------:|-------------:|-----:|---------------:|--------------------:|------------:|:--------------|:---------------------------|----------:|--------------:|--------:|
|      nan |           32 |  nan |            nan |                 nan |           0 | rgb           | jhu-clsp/ettin-encoder-17m |       nan |           nan |     nan |

## Grid keys (vary across runs)
`model`, `backbone`, `features`, `included_folders`, `meta_encoder`, `meta_layers`, `meta_text_template`

## Top Runs
| run_id       | model   | backbone   | features        | included_folders   | meta_encoder   |   meta_layers | meta_text_template   |   best_metric |   best_epoch |
|:-------------|:--------|:-----------|:----------------|:-------------------|:---------------|--------------:|:---------------------|--------------:|-------------:|
| 46d02f82f8a6 | late    | resnet18   | image+mean+meta | focus/iphone       | flair          |            -2 | compact              |        4.9181 |           20 |
| 368221ea759e | late    | resnet18   | image+mean+meta | focus/iphone       | flair          |            -2 | compact              |       12.2449 |           20 |
