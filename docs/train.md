# Training instructions

## Parameters definition

Model training is performed using the [training script](../training/train.sh).

At the beginning of the script, some parameters must be defined:

* `arch`: the model architecture to be trained (`yolov8`, `yolov12`, `faster`)
* `slug`: a unique identifier for the run (optional, default: `train`)
* `$STORAGE` env variable: path to the storage folder for the dataset and results
* `$TMPDIR` env variable: path to temporary dataset generated for each fold
* `$ENVSDIR` env variable: path to the virtual environments folder

## Dataset download

Download the DroneWaste dataset from [Zenodo](TODO_ADD_ZENODO_LINK) and extract it in the `$STORAGE/dronewaste_v1.0` folder.

## Training script

Start the model training procedure using the following command:

```bash
cd training/
bash train.sh
```

The training script will select the correct environment based on the selected model architecture.

Model training on the DroneWaste dataset is performed using a k-fold cross-validation approach where each site is treated as a separate fold.
The training script iterates over all sites. At each iteration, a temporary dataset is generated where the current site is used as the test set while the remaining sites are used as the training set.

The results from each fold training are saved in separate folders inside the `$STORAGE/kfold_results/$run_id` folder.
