# yolo3 
Yolov3 implementation in pytorch based on https://github.com/ultralytics/yolov3

## Clone the repository ##
  
We need to clone the repository recursively because of coco submodule.

```
git clone --recursive https://github.com/okanlv/yolo3
cd yolo3
```
## MSCOCO Dataset ##

If you already have the MSCOCO images, you can use symlink inside yolo3 directory as follows

```
mkdir coco/images
cd coco/images
ln -s path/to/train2014 train2014
ln -s path/to/val2014 val2014
```
Then run

```
cd ..
wget -c https://pjreddie.com/media/files/instances_train-val2014.zip
wget -c https://pjreddie.com/media/files/coco/5k.part
wget -c https://pjreddie.com/media/files/coco/trainvalno5k.part
wget -c https://pjreddie.com/media/files/coco/labels.tgz
tar xzf labels.tgz
unzip -q instances_train-val2014.zip
```

Otherwise run the following command.

```
./get_coco_dataset.sh
```

## VOC Dataset ##

If you already have the VOC Dataset, you can use symlink inside yolo3 directory as follows

```
ln -s path/to/VOCdevkit VOCdevkit
```

Otherwise run the following commands

```
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar
```

Afterwards, run `voc_label.py` to generate the following text files containing image paths

```
2007_train.txt
2007_val.txt
2007_test.txt
2012_train.txt
2012_val.txt
```

We can use one of the above files to train our model and the other one to test it. As an alternative,
following https://pjreddie.com/darknet/yolo/, we can combine everything except 2007_test.txt into one
.txt file and use it to train our model with the following code.

```
cat 2007_train.txt 2007_val.txt 2012_*.txt > train.txt
```

Modify train and valid paths in cfg/voc.data to point to your preferred train and validation text files.

## Download pretrained models ##

Inside the yolo3 directory, run

```
./download_weights.sh
```

