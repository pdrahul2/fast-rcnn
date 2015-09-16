# Train the HHA Alexnet model from the supervision transfer initialization weights
PYTHONPATH='.' PYTHONUNBUFFERED="True" python tools/train_net.py --gpu 0 \
  --solver output/training_demo_hha/solver.prototxt.hha \
  --weights data/init_models/ST_vgg_to_alexnet_hha/ST_vgg_to_alexnet_hha.caffemodel \
  --imdb nyud2_hha_2015_trainval \
  --cfg output/training_demo_hha/config.prototxt.hha \
  --iters 40000 \
  2>&1 | tee output/training_demo_hha/train.log


# Train the VGG model on the color images
PYTHONPATH='.' PYTHONUNBUFFERED="True" python tools/train_net.py --gpu 2 \
  --solver output/training_demo_rgb_vgg/solver.prototxt.images \
  --weights data/init_models/VGG16/VGG16.v2.caffemodel \
  --imdb nyud2_images_2015_trainval \
  --cfg output/training_demo_rgb_vgg/config.prototxt.images \
  2>&1 | tee output/training_demo_rgb_vgg/train.log

# Train the alexnet model on the color images
PYTHONPATH='.' PYTHONUNBUFFERED="True" python tools/train_net.py --gpu 0 \
  --solver output/training_demo_rgb_alexnet/solver.prototxt.images \
  --weights data/init_models/CaffeNet/CaffeNet.v2.caffemodel \
  --imdb nyud2_images_2015_trainval \
  --cfg output/training_demo_rgb_alexnet/config.prototxt.images \
  2>&1 | tee output/training_demo_rgb_alexnet/train.log


