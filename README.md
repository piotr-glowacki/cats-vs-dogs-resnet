# Cats vs Dogs Binary Classification
This repository contains code and resources for training a model that distinguishes between cat and a dog on images from [Cats-vs-Dogs](https://huggingface.co/datasets/cats_vs_dogs). The model aims to accurately identify if the given image represents a cat or a dog.

## Getting started
### 1. Clone this repository:
```bash
git clone https://github.com/piotr-glowacki/cats-vs-dogs-resnet.git
```
### 2. Download the data:
```bash
wget https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip
```
### 3. Unzip the downloaded file:
```bash
unzip kagglecatsanddogs_5340.zip
```
### 4. Install dependencies:
```bash
pip install -r requirements.txt
```
### 5. Split the data into train/val/test:
```bash
python3 train_test_splitter.py
```

## Usage
### 1. Run train.py file:
```bash
python3 train.py
```
By default training takes the arguments as below:
* lr = 0.001,
* momentum = 0.92,
* epochs = 50,
* batch_size = 16,
* num_workers = 4,
* criterion = nn.CrossEntropyLoss(),
* model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
* onnx = True
Executing the command will do the following:
* Initialize ResNet-18 model from torchvision.models.
* Change the last layer to output only two neurons, which represent the two categories ("0", "1").
* Train the model with given specification.
* Export model's state to .pth format.
* Export model to ONNX format if 'onnx' parameter is set to True.
### 2. Run test.py file:
```bash
python3 test.py
```
Executing the command will do the following:
* 
