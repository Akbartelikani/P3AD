#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch


def predict_bounding_boxes(image_matrix, faster_rcnn_model):
    # Assume 'faster_rcnn_model' is a trained Faster R-CNN model in PyTorch
    with torch.no_grad():
        # Convert 'image_matrix' to a PyTorch tensor
        image_tensor = torch.from_numpy(image_matrix).unsqueeze(0).permute(0, 3, 1, 2).float()
        

        # Forward pass to get the model predictions
        predictions = faster_rcnn_model(image_tensor)

    # Extract the bounding box predictions from the model output
    predicted_boxes = predictions[0]['boxes'].numpy()

    return predicted_boxes

def loss_function(original_image,adversarial_image,model):

    predicted_boxes_original, predicted_scores_original = model.predict(original_image)
    predicted_boxes_adversarial, predicted_scores_adversarial = model.predict(adversarial_image)

    # Loss for bounding box predictions
    localization_loss = np.sum((predict_bounding_boxes(original_image) - predict_bounding_boxes(adversarial_image))**2)

    # Loss for classification predictions
    classification_loss = -np.sum(predicted_scores_original * np.log(predicted_scores_adversarial))

    # Combine the losses
    total_loss = localization_loss + classification_loss

    return total_loss


def pixel_set_selection(image_matrix, model):
    rows, cols, channels = image_matrix.shape
    threshold_scale=1.0
    
    critical_pixels = []
    for i in range(rows):
        for j in range(cols):
            for k in range(channels):
                # Compute gradient using finite differences
                delta = 1e-5
                perturbed_image = image_matrix.copy()
                perturbed_image[i, j, k] += delta
                gradient = (loss_function(image_matrix,perturbed_image, model)) / delta

                # Compute magnitude of the gradient
                magnitude = np.sqrt(np.sum(gradient**2))
                
                # Check if the magnitude surpasses the threshold
                mean_magnitude = np.mean(magnitude)
                threshold = threshold_scale * mean_magnitude

                # Check if the magnitude surpasses the threshold
                if magnitude > threshold:
                    critical_pixels.append((i, j))

    return critical_pixels

