#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from PIL import Image
import numpy as np
import Perturbation_Generation as PG
import Pixel_set_selection as PSS
import Frame_Selection as FS

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch




def build_faster_rcnn_model(input_shape=(None, None, 3), num_classes=21):
    # Backbone (Feature Extractor)
    backbone = tf.keras.applications.ResNet50(
        include_top=False,
        input_shape=input_shape,
        weights='imagenet'
    )

    # Region Proposal Network (RPN)
    rpn = keras.models.Sequential([
        layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(1, (1, 1), activation='sigmoid')
    ])

    # Region of Interest (RoI) Pooling
    roi_pooling = keras.layers.RoiPoolingConv(7, 7)([backbone.output, rpn.output])

    # Fully Connected layers for classification and bounding box regression
    flatten = layers.Flatten()(roi_pooling)
    fc1 = layers.Dense(512, activation='relu')(flatten)
    fc2_cls = layers.Dense(num_classes, activation='softmax', name='output_class')(fc1)
    fc2_reg = layers.Dense(num_classes * 4, activation='linear', name='output_bbox')(fc1)

    # Final Faster R-CNN model
    faster_rcnn_model = keras.models.Model(inputs=backbone.input, outputs=[fc2_cls, fc2_reg])

    return faster_rcnn_model



def P3AD(extracted_frames):
    num_iterations=30
    Step size=0.01
    Perturbation magnitude=0.01
    
    # Create an instance of the Faster R-CNN model
    faster_rcnn_model = build_faster_rcnn_model()
    for period_frames_list in frames:
        for frame in period_frames_list:
            
            # Apply Bayesian optimization to get an image matrix
            image_matrix_selected = FS.bayesian_optimization(frame, max_iterations, theta_initial, alpha, beta1, beta2, epsilon)

            # Select critical pixels using Faster R-CNN
            pixel_matrix_selected = PSS.pixel_set_selection(image_matrix_selected, faster_rcnn_model)

            # Perturb the selected pixels in the image matrix
            perturbed_frame = perturbation_generation(image_matrix_selected, region_proposals, num_iterations, step_size, perturbation_magnitude, pixel_matrix_selected)

            # Replace perturbed_frame with the original frame
            frame[:] = image_matrix_selected[0]
        

