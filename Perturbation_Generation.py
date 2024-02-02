#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50

def faster_rcnn_resnet_model(input_shape=(None, None, 3), num_classes=21):
    # Load pre-trained ResNet50 model as the backbone
    resnet_backbone = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Region Proposal Network (RPN)
    rpn = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(resnet_backbone.output)
    rpn_class = layers.Conv2D(2, (1, 1), activation='sigmoid', name='rpn_class')(rpn)
    rpn_bbox = layers.Conv2D(4, (1, 1), name='rpn_bbox')(rpn)

    # Create the region of interest (RoI) pooling layer
    roi_pooling = layers.RoiPoolingConv(7, 7)([resnet_backbone.output, rpn_bbox])

    # Fully Connected layers for classification and bounding box regression
    flatten = layers.Flatten()(roi_pooling)
    fc1 = layers.Dense(512, activation='relu')(flatten)
    fc2_cls = layers.Dense(num_classes, activation='softmax', name='output_class')(fc1)
    fc2_reg = layers.Dense(num_classes * 4, activation='linear', name='output_bbox')(fc1)

    # Combine all components into the final Faster R-CNN model
    faster_rcnn_model = tf.keras.Model(inputs=resnet_backbone.input, outputs=[fc2_cls, fc2_reg, rpn_class, rpn_bbox])

    return faster_rcnn_model

def RegionPooling(x, region_proposals, perturbation_mask):
    # Convert image to PyTorch tensor
    transform = T.Compose([T.ToTensor()])
    x_tensor = transform(x).unsqueeze(0)

    # Convert region proposals to PyTorch tensor
    region_proposals_tensor = torch.tensor(region_proposals, dtype=torch.float32).unsqueeze(0)

    # Convert perturbation mask to PyTorch tensor
    perturbation_mask_tensor = torch.tensor(perturbation_mask, dtype=torch.float32).unsqueeze(0)

    model = faster_rcnn_resnet_model(pretrained=True)
    model.eval()

    with torch.set_grad_enabled(True):
        x_tensor.requires_grad_(True)
        region_proposals_tensor.requires_grad_(True)
        perturbation_mask_tensor.requires_grad_(True)

        # Forward pass through the Faster R-CNN model
        output = model(x_tensor, [region_proposals_tensor])

        # Assuming the loss is computed based on the model's output
        loss = torch.sum(output['boxes'])

        # Backward pass to compute the gradients
        loss.backward()

        # Update the image tensor using the perturbation mask
        perturbed_x_tensor = x_tensor + perturbation_mask_tensor

    # Convert the perturbed image tensor back to a NumPy array
    perturbed_x = perturbed_x_tensor.squeeze(0).permute(1, 2, 0).detach().numpy()

    return perturbed_x


def perturbation_generation(x, region_proposals, num_iterations, step_size, perturbation_magnitude, critical_pixels):
    perturbed_frame = x.copy()
    
    model=faster_rcnn_resnet_model()
    # Assuming 'x' is your input image (you may need to preprocess it)
    x = tf.keras.preprocessing.image.img_to_array(x)
    x = tf.keras.applications.resnet50.preprocess_input(x[tf.newaxis, ...])

    # Perform inference to get region proposals
    predictions = model.predict(x)

    # Extract region proposals
    region_proposals = predictions[-1]  # Assuming the last element of predictions is the rpn_bbox


    perturbed_region_proposals = region_proposals.copy()
    
    
    
    for _ in range(num_iterations):
        # Compute the gradient of the loss with respect to the perturbed image
        gradient_x = ComputeGradient(perturbed_frame, perturbed_region_proposals)

        # Update the region features based on the gradient information
        perturbed_region_proposals += step_size * gradient_x

        # Compute the perturbed image based on the updated region features and the pixel mask
        perturbed_frame = RegionPooling(perturbed_frame, perturbed_region_proposals, perturbation_magnitude * critical_pixels * gradient_x)

    return perturbed_frame

