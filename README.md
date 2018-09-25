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
## Download pretrained models ##

Inside the yolo3 directory, run

```
./download_weights.sh
```

