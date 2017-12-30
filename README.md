# SegLink

Detecting Oriented Text in Natural Images by Linking Segments (https://arxiv.org/abs/1703.06520).

## Prerequisites

The project is written in Python3 and C++ and relies on TensorFlow v1.0 or newer. We have only tested it on Ubuntu 14.04. If you are using other Linux versions, we suggest using Docker. CMake (version >= 2.8) is required to compile the C++ code. Install TensorFlow (GPU-enabled) by following the instructions on https://www.tensorflow.org/install/. The project requires no other Python packages.

On Ubuntu 14.04, install the required packages by
```
sudo apt-get install cmake
sudo pip install --upgrade tensorflow-gpu
```

## Installation

The project uses `manage.py` to execute commands for compiling code and running training and testing programs. For installation, execute
```
./manage.py build_op
```
in the project directory to compile the custom TensorFlow operators written in C++. To remove the compiled binaries, execute
```
./manage.py clean_op
```

## Dataset Preparation

See ``tool/create_datasets.py''

## Training

```
./manage.py <exp-directory> train
```

## Evaluation

See ``evaluate.py''
