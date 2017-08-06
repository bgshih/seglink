CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(seglink)

# compiler flags
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ${OpenMP_CXX_FLAGS} -Wall -fPIC -D_GLIBCXX_USE_CXX11_ABI=0")

# TensorFlow dependencies
EXECUTE_PROCESS(COMMAND python3 -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; import tensorflow as tf; print(tf.sysconfig.get_include())"
                OUTPUT_VARIABLE TF_INC)
MESSAGE(STATUS "Found TF_INC: " ${TF_INC})

# boost
# find_package (Boost REQUIRED COMPONENTS graph)

# target
INCLUDE_DIRECTORIES(${TF_INC})
ADD_LIBRARY(seglink SHARED 
  utilities.h
  sample_crop_bbox_op.cc
  encode_groundtruth_op.cc
  decode_segments_links_op.cc
  combine_segments_op.cc
  detection_mask_op.cc
  clip_rboxes_op.cc
  polygons_to_rboxes_op.cc
  project_polygons_op.cc)
