# Sweep Summary (monitor: `de00`, mode: `min`)

Total unique runs: **2**
Best run metric (test): **6.3098**

## Constants
|   epochs |   batch_size | color_space   |   optim |   lr |   weight_decay |   lr_schedule |   lr_auto |   mse_weight_epochs |   val_split | meta_model_name            |
|---------:|-------------:|:--------------|--------:|-----:|---------------:|--------------:|----------:|--------------------:|------------:|:---------------------------|
|      nan |           32 | rgb           |     nan |  nan |            nan |           nan |       nan |                 nan |           0 | jhu-clsp/ettin-encoder-17m |

## Grid keys (vary across runs)
`model`, `backbone`, `features`, `included_folders`, `meta_encoder`, `meta_layers`, `meta_text_template`

## Top Runs
| run_id       |   epochs |   batch_size | color_space   | optim   |    lr |   weight_decay | lr_schedule   | lr_auto   |   mse_weight_epochs | model   | backbone   | features        | included_folders   | meta_encoder   |   meta_layers | meta_text_template   |   best_metric |   best_epoch |
|:-------------|---------:|-------------:|:--------------|:--------|------:|---------------:|:--------------|:----------|--------------------:|:--------|:-----------|:----------------|:-------------------|:---------------|--------------:|:---------------------|--------------:|-------------:|
| 46d02f82f8a6 |       20 |           32 | rgb           | sgd     | 0.001 |         0.0001 | paper_resnet  | True      |                  75 | late    | resnet18   | image+mean+meta | focus/iphone       | flair          |            -2 | compact              |       6.30983 |           20 |
