# knn-project
Project in KNN course at BUT FIT

# Setup Detector v1

## Dependencies
```
pip install -U torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html
pip install cython pyyaml==5.1
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install detectron2==0.1.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html
pip install opencv-python
```

# Setup Detector v2

## Dependencies
Install latest (1.8) PyTorch (and TorchVision) from: https://pytorch.org/get-started/locally/.
```
pip install pyyaml==5.1
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.8/index.html
pip install opencv-python
```

# Troubleshooting
In case of `error: invalid command 'bdist_wheel'` (via [Stack Overflow](https://stackoverflow.com/questions/34819221/why-is-python-setup-py-saying-invalid-command-bdist-wheel-on-travis-ci):
```
/usr/bin/python3 -m pip install --upgrade pip
sudo apt install gcc libpq-dev -y
sudo apt install python3-dev python3-pip python3-venv python3-wheel -y
pip3 install wheel
```
