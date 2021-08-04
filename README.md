# Drivable area segmentation

The repository is used to train a drivable-area-segmentation model with deeplabv3+, and the trained model can inference with libtorch. The repository is based on [deeplabs](https://github.com/sunggukcha/deeplabs) and [Person_Segmentation](https://github.com/runrunrun1994/Person_Segmentation).

## Data Preparation
Please prepare dataset under the **data** directory. You can download from link：https://pan.baidu.com/s/1NT5-Ldo0kYIfCcUbkq0xEA password：76bi

* Generate labels for **custom dataset**. The test samples are under  the **samples** directory, and generated labels are under the **results** directory.
```
python hand_seg.py 
```

## Pretrained model
Please download pretrained model under the **weights** directory. Link：https://pan.baidu.com/s/1l78GrdJ8FoghqfzGGw0Kng 
password：vm33

## Train
```
python train.py 
```

## Inference with python
You download the trained model from link(https://pan.baidu.com/s/1rnOtdmewT-oho-cU6bLluw password：djue)  and place it under the **weights** directory. The test samples are under  the **samples** directory, and inferece results are under the **results** directory.
```
python inference.py 
```

## Inference with c++
```
python generate_half_model.py
cd  LibtorchSegmentation/build
cmake ..
make
./drive
```