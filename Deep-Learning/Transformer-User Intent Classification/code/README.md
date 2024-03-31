[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/AL4k43eR)
# PA4 - Transformers for Amazon intent classification

## Installation
```
pip install -r requirements.txt
```

## To run the code
Edit or uncomment the command you wish to run in `run.sh`, then run the command `bash run.sh` in your terminal to train the model, get test accuracy, and have the configurations and results all recorded in the `results` folder of the directory. 

In the following section, we provide a sample command line with our best configurations for each of our 3 models that we experimented with in this PA (baseline, custom models and contrastive learning). 

### Baseline model
```
python main.py --n-epochs 10 --batch-size 64 --learning-rate 0.0001 --hidden-dim 250 --drop-rate 0.5 --do-train
```

### Custom model
For custom Model, the default would be Stochastic Weight Averaging (SWA). The model would apply SWA after the epoch number > swa_start (default 1) due to fast convergence of model. To apply Layer-wise Learning Rate Decay technique (LLRD), an additional argument --advanced 0 needs to be passed in. By doing so, different learning rate would be applied to different layers. The learning rate would be mulitplied with lr_multi (default 0.95). 

The result is apply LLRD helps to improve performance, but SWA and combined method falls short to the inital assumption. All the techniques has been tuned for best hyperparameters like lr_multi, swa_start, swa_lr,drop_out rate, learning rate, and etc.
```
python main.py --hidden-dim 150 --debug --task custom --do-train --drop-rate 0.5
```

### Contrastive Learning
```
python main.py --hidden-dim 150 --debug --task supcon --adam-epsilon 1e-8 --CL-hidden-size 256 --n-epochs-CL-CLS 50 --CL-Mask 1 --do-train --drop-rate 0.2 --max-len 20 --cl-learning "SimCLR" --n-epochs 20 --batch-size 16
```

Specify whether you want to use the `SimCLR` Loss or `SupCon` loss for training for the `cl-learning` parameter. 
