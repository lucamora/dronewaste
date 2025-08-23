# Evaluation instructions

The [evaluation notebook](../evaluation/evaluation.ipynb) is based on a [forked repo](https://github.com/wtiandong/tide) of the original [TIDE toolbox](https://github.com/dbolya/tide) which was adapted to report per-class APs.

The [evaluation notebook](../evaluation/evaluation.ipynb) collects the predictions for all folds and evaluates the performance against the global ground truth.
Model performance is evaluated by computing the AP for each category and the mean AP using an IOU threshold of 0.5.

## Variables definition

At the beginning of the notebook, some variables must be defined:

* `SPLIT`: the split to evaluate (`test` or `valid`)
* `PREDICTIONS_PATH`: path to the folder containing the results of the selected run
* `GT_PATH`: path to the global ground truth
