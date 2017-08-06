# convert caffe model
CUDA_VISIBLE_DEVICES=3 python dump_caffemodel_weights.py \
  --caffe_root /home/bgshi/research/common/caffe/ \
  --prototxt_path /home/bgshi/research/object_detection/ssd.caffe/models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced_deploy.prototxt \
  --caffemodel_path /home/bgshi/research/object_detection/ssd.caffe/models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel \
  --caffe_weights_path ./VGG_ILSVRC_16_layers_weights.pkl

# # convert to tensorflow checkpoint
CUDA_VISIBLE_DEVICES=3 python convert_caffemodel_to_ckpt.py \
  --model_scope ssd/vgg16 \
  --ckpt_path ../../model/VGG_ILSVRC_16_layers_ssd.ckpt \
  --caffe_weights_path ./VGG_ILSVRC_16_layers_weights.pkl
