# Sample Image Classification
An implementation of image classification based on ResNet

## Requirement
- Python 3.6.5
- Tensorflow 1.10.1 
- Pillow 5.0.0
- numpy 1.14.5
- opencv-python 3.4.3.18

## Dataset
- [STL-10 dataset](https://cs.stanford.edu/~acoates/stl10/)

## Training
1) Download [STL-10 dataset](https://cs.stanford.edu/~acoates/stl10/)

2) Unzip the dataset zip file
```
tar -xvf ./path/to/file/stl10_binary.tar.gz
```

3) Train the model
```
python main.py --tr_img ./path/to/file/test_X.bin --tr_label ./path/to/file/test_y.bin --te_img ./path/to/file/train_X.bin --te_label ./path/to/file/train_y.bin
```

## Test
```
python main.py --mode test --pre_trained_model ./model/model_best_on_val --te_img ./path/to/file/train_X.bin --te_label ./path/to/file/train_y.bin
```

## Making grad cam
```
python main.py --mode grad_cam --pre_trained_model ./model/model_best_on_val --te_img ./path/to/file/train_X.bin --te_label ./path/to/file/train_y.bin
```

## Details
- The number of parameters : 2,952,928 (less than 3M)
- Top 1 accuracy(test data) : 78.3%
- I used learning rate scheduling
(49-th line in res_model.py)
- I used two data augmentation(The functions (data_aug(), extract_patch() in util.py)

1) With random horizontal flip 
2) With single scale jittering

- I used separate training set, validataion set, and test set
(The function split_data() in util.py)

## Experimental Results
<img src = "images/train_acc.png" height = "250px">
<img src = "images/train_loss.png" height = "250px">
For each train batch, these are the train accuracy and train loss graphs 

If you want to see these results in tensorboard,
```
tensorboard --logdir=./log
```

log.txt in log directory : Training log file, which measured every epoch for total train data and validation data 

## Grad Cam Results
옳게 예측된 데이터

| Input | Grad cam |
| --- | --- |
| <img src="images/real_data/True_0160.png"> |<img src="images/grad_cam/True_0160.png">| 
| <img src="images/real_data/True_0521.png"> |<img src="images/grad_cam/True_0521.png">|
| <img src="images/real_data/True_2163.png"> |<img src="images/grad_cam/True_2163.png">|

틀리게 예측된 데이터

| Input | Grad cam |
| --- | --- |
| <img src="images/real_data/False_0657.png"> |<img src="images/grad_cam/False_0657.png">| 
| <img src="images/real_data/False_1457.png"> |<img src="images/grad_cam/False_1457.png">|
| <img src="images/real_data/False_2584.png"> |<img src="images/grad_cam/False_2584.png">|

분석 : 올바르게 예측한 데이터의 경우, model이 image를 예측하기 적합한 곳에 강한 activation을 주었음을 알 수 있다. 예를들어, 동물의 몸통을 포함하는 전체적인 부분이나 동물의 얼굴부분에 강한 activation을 주었음을 위의 데이터를 통해 알 수 있다. 틀리게 예측한 데이터에 대해서는 model이 배경이나, 몸통의 극히 일부 혹은 classification에 도움되지않는 발 끝부분 등에 강한 activation을 주었음을 알 수 있다.

## Reference
[1] https://github.com/mttk/STL10

- The functions read_labels and read_all_images in util.py are borrowed from [1]

[2] https://github.com/cydonia999/Grad-CAM-in-TensorFlow

- The function save_cam in util.py is borrowed from [2]
