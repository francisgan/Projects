# DSC-180B-Hardware-Acceleration

## Table of Contents
- [DSC-180B-Hardware-Acceleration](#dsc-180b-hardware-acceleration)
  - [Table of Contents](#table-of-contents)
  - [Environment Setup](#environment-setup)
  - [Data Acquisition](##Data)
  - [Models](##Models)
  - [Simulation and Profiling](##Simulation)
  - [Config File](##Config)

## Environment Setup

Please refer to the [Setup Guide](docs/Setup.md) for instructions on how to build and run the Docker image and container.

### Python Environment Requirement

#### Version
- Python 3.9 or higher

#### Required Libraries
- scikit-learn
- numpy
- Pytorch (Local) - Note: We need access to a GPU for PyTorch; otherwise, set `enable_GPU=False` in the `config.py` file. The CUDA version should match the PyTorch version. Specific information can be found at [PyTorch Getting Started](https://pytorch.org/get-started/locally/).
- torchvision

## Data
The dataset is too large to fit in GitHub. The original dataset can be downloaded from [PAMAP2 Physical Activity Monitoring](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring). Unzip the `PAMAP2_Dataset` you downloaded, which includes a `readme.pdf` with a specific description of the data. Then, copy and paste the `Optional` and `Protocol` directories to `/data` in our repository. Thus, in `/data/Optional`, there should be 5 files called `subject1xx.dat`; and 9 files in `/data/Protocol` called `subject1xx.dat`. Run `python generate_data.py`, which will generate three files: `small_sample`, `feature`, and `target`, which are cleaned and ready for deep learning models. The `small_sample` file contains the first 10 examples of data, which can be read by code faster and will be used in the simulation step.

Note: `small_sample` file is already in the repository. The above steps make the data generating process reproducible. `outfile` and `target` are too large and take times to generate, __an alternative way__ to get them is to download from google drive([target](https://drive.google.com/file/d/1dURM_EeSsQ9zuvCBRaBhG4nP1AxXnYmg/view) and [feature](https://drive.google.com/file/d/1zIjJrfmqwadWeDPuVVgf0pbgxxUF2xWp/view?usp=drive_link)) ,and put them in root directory

## Models
Run `python xxx_train.py` to train the corresponding model, print the model accuracy on the test set, and save the model in `/models`.

Note: Training models take a long time, you can reproduce the models by running above code. Or, you can download models [here](https://drive.google.com/drive/folders/1_sqHDKapqrQPw_6xNoGvAgTEd3-KmeGB?usp=drive_link), and put them in `/models`.
Or run this command to download large model `python download_model.py`, this command require `gdown` installed in python environment.

## Simulation
In real-life situation, a trained model is saved on a device. Sensors on device store data on a hard disk. Model will read data on memory or harh disk and then input sequences of data into the model, then the model predicts the current human activity.

We use the PyTorch profiler to track the above program operations and save the profiling data in `/profiling/environmentName/modelName/`. The JSON file can be loaded by Chrome tracer (`chrome://tracing`) and generate a visualization.

Use following command to simulate all model on current environment.
```
python run_simulation.py --env "environment name" --gpu "True/False" --other "True/False" --trial #int
```
For example, current environment is "env10" with GPU, and it's third time we run it:
```
python run_simulation.py --env env10 --gpu True --other False --trial 3
```

## Config
Data generating parameters and model parameters are stored in `configs.py`. Details are explained in the config file.
