### Download models and data
1. Download the NYUD2 data

  ```Shell
  cd $FRCN_ROOT
  ./data/scripts/fetch_nyud2_data.sh
	```
	
2. Download the NYUD2 MCG boxes

	```Shell
  ./data/scripts/fetch_nyud2_mcg_boxes.sh
	```

3. Download the ImageNet and Supervision Transfer Models 

	```Shell
  ./data/scripts/fetch_init_models.sh
  ```

3. Fetch NYUD2 Object Detector Models.

	```Shell
  ./outputs/fetch_nyud2_models.sh
  ```

### Usage

**Train** a Fast R-CNN detector.

```Shell
./tools/train_net.py --gpu 0 --solver models/VGG16/solver.prototxt \
	--weights data/imagenet_models/VGG16.v2.caffemodel
```

If you see this error

```
EnvironmentError: MATLAB command 'matlab' not found. Please add 'matlab' to your PATH.
```

then you need to make sure the `matlab` binary is in your `$PATH`. MATLAB is currently required for PASCAL VOC evaluation.

**Test** a Fast R-CNN detector. For example, test the VGG 16 network on VOC 2007 test:

```Shell
./tools/test_net.py --gpu 1 --def models/VGG16/test.prototxt \
	--net output/default/voc_2007_trainval/vgg16_fast_rcnn_iter_40000.caffemodel
```

Test output is written underneath `$FRCN_ROOT/output`.

**Compress** a Fast R-CNN model using truncated SVD on the fully-connected layers:

```Shell
./tools/compress_net.py --def models/VGG16/test.prototxt \
	--def-svd models/VGG16/compressed/test.prototxt \
    --net output/default/voc_2007_trainval/vgg16_fast_rcnn_iter_40000.caffemodel
# Test the model you just compressed
./tools/test_net.py --gpu 0 --def models/VGG16/compressed/test.prototxt \
	--net output/default/voc_2007_trainval/vgg16_fast_rcnn_iter_40000_svd_fc6_1024_fc7_256.caffemodel
```


