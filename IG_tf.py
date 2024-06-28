# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 16:10:55 2024

@author: Saleh
"""

import matplotlib.pylab as plt
import numpy as np
import math
import sys
import tensorflow as tf

def generate_alphas(m_steps=50,
                    method='riemann_trapezoidal'):
  """
  Args:
    m_steps(Tensor): A 0D tensor of an int corresponding to the number of linear
      interpolation steps for computing an approximate integral. Default is 50.
    method(str): A string representing the integral approximation method. The 
      following methods are implemented:
      - riemann_trapezoidal(default)
      - riemann_left
      - riemann_midpoint
      - riemann_right
  Returns:
    alphas(Tensor): A 1D tensor of uniformly spaced floats with the shape 
      (m_steps,).
  """
  m_steps_float = tf.cast(m_steps, float) # cast to float for division operations.

  if method == 'riemann_trapezoidal':
    alphas = tf.linspace(0.0, 1.0, m_steps+1) # needed to make m_steps intervals.
  elif method == 'riemann_left':
    alphas = tf.linspace(0.0, 1.0 - (1.0 / m_steps_float), m_steps)
  elif method == 'riemann_midpoint':
    alphas = tf.linspace(1.0 / (2.0 * m_steps_float), 1.0 - 1.0 / (2.0 * m_steps_float), m_steps)
  elif method == 'riemann_right':    
    alphas = tf.linspace(1.0 / m_steps_float, 1.0, m_steps)
  else:
    raise AssertionError("Provided Riemann approximation method is not valid.")

  return alphas

def integral_approximation(gradients, 
                           method='riemann_trapezoidal'):
  """Compute numerical approximation of integral from gradients.

  Args:
    gradients(Tensor): A 4D tensor of floats with the shape 
      (m_steps, img_height, img_width, 3).
    method(str): A string representing the integral approximation method. The 
      following methods are implemented:
      - riemann_trapezoidal(default)
      - riemann_left
      - riemann_midpoint
      - riemann_right 
  Returns:
    integrated_gradients(Tensor): A 3D tensor of floats with the shape
      (img_height, img_width, 3).
  """
  if method == 'riemann_trapezoidal':  
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
  elif method == 'riemann_left':
    grads = gradients
  elif method == 'riemann_midpoint':
    grads = gradients
  elif method == 'riemann_right':    
    grads = gradients
  else:
    raise AssertionError("Provided Riemann approximation method is not valid.")

  # Average integration approximation.
  integrated_gradients = tf.math.reduce_mean(grads, axis=0)


  return integrated_gradients

def generate_path_inputs(baseline,
                         input_img,
                         alphas):
  """Generate m interpolated inputs between baseline and input features.
  Args:
    baseline(Tensor): A 3D image tensor of floats with the shape 
      (img_height, img_width, 3).
    input(Tensor): A 3D image tensor of floats with the shape 
      (img_height, img_width, 3).
    alphas(Tensor): A 1D tensor of uniformly spaced floats with the shape 
      (m_steps,).
  Returns:
    path_inputs(Tensor): A 4D tensor of floats with the shape 
      (m_steps, img_height, img_width, 3).
  """
  # Expand dimensions for vectorized computation of interpolations.
  alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
  baseline_x = tf.expand_dims(baseline, axis=0)
  input_x = tf.expand_dims(input_img, axis=0) 
  delta = input_x - baseline_x
  path_inputs = baseline_x +  alphas_x * delta
  
  return path_inputs

def convergence_check(model, attributions, baseline, input_img, target_class_idx):
  """
  Args:
    model(keras.Model): A trained model to generate predictions and inspect.
    baseline(Tensor): A 3D image tensor with the shape 
      (image_height, image_width, 3) with the same shape as the input tensor.
    input(Tensor): A 3D image tensor with the shape 
      (image_height, image_width, 3).
    target_class_idx(Tensor): An integer that corresponds to the correct 
      ImageNet class index in the model's output predictions tensor. Default 
        value is 50 steps.   
  Returns:
    (none): Prints scores and convergence delta to sys.stdout.
  """
  # Your model's prediction on the baseline tensor. Ideally, the baseline score
  # should be close to zero.
  baseline_prediction = model(tf.expand_dims(baseline, 0))
  baseline_score = tf.nn.softmax(tf.squeeze(baseline_prediction))[target_class_idx]
  # Your model's prediction and score on the input tensor.
  input_prediction = model(tf.expand_dims(input_img, 0))
  input_score = tf.nn.softmax(tf.squeeze(input_prediction))[target_class_idx]
  # Sum of your IG prediction attributions.
  ig_score = tf.math.reduce_sum(attributions)
  delta = ig_score - (input_score - baseline_score)
  try:
    # Test your IG score is <= 5% of the input minus baseline score.
    tf.debugging.assert_near(ig_score, (input_score - baseline_score), rtol=0.05)
    tf.print('Approximation accuracy within 5%.', output_stream=sys.stdout)
  except tf.errors.InvalidArgumentError:
    tf.print('Increase or decrease m_steps to increase approximation accuracy.', output_stream=sys.stdout)
  
  tf.print('Baseline score: {:.3f}'.format(baseline_score))
  tf.print('Input score: {:.3f}'.format(input_score))
  tf.print('IG score: {:.3f}'.format(ig_score))     
  tf.print('Convergence delta: {:.3f}'.format(delta))

def compute_gradients(model, path_inputs, target_class_idx):
  """Compute gradients of model predicted probabilties with respect to inputs.
  Args:
    mode(tf.keras.Model): Trained Keras model.
    path_inputs(Tensor): A 4D tensor of floats with the shape 
      (m_steps, img_height, img_width, 3).
    target_class_idx(Tensor): A 0D tensor of an int corresponding to the correct
      ImageNet target class index.
  Returns:
    gradients(Tensor): A 4D tensor of floats with the shape 
      (m_steps, img_height, img_width, 3).
  """
  with tf.GradientTape() as tape:
    tape.watch(path_inputs)
    predictions = model(path_inputs)
    # Note: IG requires softmax probabilities; converting Inception V1 logits.
    outputs = tf.nn.softmax(predictions, axis=-1)[:, target_class_idx]      
  gradients = tape.gradient(outputs, path_inputs)

  return gradients

@tf.function
def integrated_gradients(model,
                         baseline, 
                         input_img,  
                         target_class_idx,
                         m_steps=50,
                         method='riemann_trapezoidal',
                         batch_size=32
                        ):
  """
  Args:
    model(keras.Model): A trained model to generate predictions and inspect.
    baseline(Tensor): A 3D image tensor with the shape 
      (image_height, image_width, 3) with the same shape as the input tensor.
    input(Tensor): A 3D image tensor with the shape 
      (image_height, image_width, 3).
    target_class_idx(Tensor): An integer that corresponds to the correct 
      ImageNet class index in the model's output predictions tensor. Default 
        value is 50 steps.           
    m_steps(Tensor): A 0D tensor of an integer corresponding to the number of 
      linear interpolation steps for computing an approximate integral.
    method(str): A string representing the integral approximation method. The 
      following methods are implemented:
      - riemann_trapezoidal(default)
      - riemann_left
      - riemann_midpoint
      - riemann_right
    batch_size(Tensor): A 0D tensor of an integer corresponding to a batch
      size for alpha to scale computation and prevent OOM errors. Note: needs to
      be tf.int64 and shoud be < m_steps. Default value is 32.      
  Returns:
    integrated_gradients(Tensor): A 3D tensor of floats with the same 
      shape as the input tensor (image_height, image_width, 3).
  """

  # 1. Generate alphas.
  alphas = generate_alphas(m_steps=m_steps,
                           method=method)

  # Initialize TensorArray outside loop to collect gradients. Note: this data structure
  # is similar to a Python list but more performant and supports backpropogation.
  # See https://www.tensorflow.org/api_docs/python/tf/TensorArray for additional details.
  gradient_batches = tf.TensorArray(tf.float32, size=m_steps+1)

  # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
  # Note: this implementation opted for lightweight tf.range iteration with @tf.function.
  # Alternatively, you could also use tf.data, which adds performance overhead for the IG 
  # algorithm but provides more functionality for working with tensors and image data pipelines.
  for alpha in tf.range(0, len(alphas), batch_size):
    from_ = alpha
    to = tf.minimum(from_ + batch_size, len(alphas))
    alpha_batch = alphas[from_:to]

    # 2. Generate interpolated inputs between baseline and input.
    interpolated_path_input_batch = generate_path_inputs(baseline=baseline,
                                                         input_img=input_img,
                                                         alphas=alpha_batch)

    # 3. Compute gradients between model outputs and interpolated inputs.
    gradient_batch = compute_gradients(model=model,
                                       path_inputs=interpolated_path_input_batch,
                                       target_class_idx=target_class_idx)
    
    # Write batch indices and gradients to TensorArray. Note: writing batch indices with
    # scatter() allows for uneven batch sizes. Note: this operation is similar to a Python list extend().
    # See https://www.tensorflow.org/api_docs/python/tf/TensorArray#scatter for additional details.
    gradient_batches = gradient_batches.scatter(tf.range(from_, to), gradient_batch)    
  
  # Stack path gradients together row-wise into single tensor.
  total_gradients = gradient_batches.stack()
    
  # 4. Integral approximation through averaging gradients.
  avg_gradients = integral_approximation(gradients=total_gradients,
                                         method=method)
    
  # 5. Scale integrated gradients with respect to input.
  integrated_gradients = (input_img - baseline) * avg_gradients

  return integrated_gradients

def plot_img_attributions(model,
                          baseline,                          
                          img,  
                          target_class_idx,
                          m_steps=50,                           
                          cmap=None,
                          overlay_alpha=0.4):
  """
  Args:
    model(keras.Model): A trained model to generate predictions and inspect.
    baseline(Tensor): A 3D image tensor with the shape 
      (image_height, image_width, 3) with the same shape as the input tensor.
    img(Tensor): A 3D image tensor with the shape 
      (image_height, image_width, 3).
    target_class_idx(Tensor): An integer that corresponds to the correct 
      ImageNet class index in the model's output predictions tensor. Default 
        value is 50 steps.
    m_steps(Tensor): A 0D tensor of an integer corresponding to the number of 
      linear interpolation steps for computing an approximate integral.
    cmap(matplotlib.cm): Defaults to None. Reference for colormap options -
      https://matplotlib.org/3.2.1/tutorials/colors/colormaps.html. Interesting
      options to try are None and high contrast 'inferno'.
    overlay_alpha(float): A float between 0 and 1 that represents the intensity
      of the original image overlay.    
  Returns:
    fig(matplotlib.pyplot.figure): fig object to utilize for displaying, saving 
      plots.
  """
  # Attributions
  ig_attributions = integrated_gradients(model=model,
                          baseline=baseline,
                          input_img=img,
                          target_class_idx=target_class_idx,
                          m_steps=m_steps)

  convergence_check(model, ig_attributions, baseline, img, target_class_idx)
  
  # Per the original paper, take the absolute sum of the attributions across 
  # color channels for visualization. The attribution mask shape is a greyscale image
  # with shape (224, 224).
  attribution_mask = tf.reduce_sum(tf.math.abs(ig_attributions), axis=-1)

  # Visualization
  fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(8, 8))

  axs[0,0].set_title('Baseline Image')
  axs[0,0].imshow(baseline)
  axs[0,0].axis('off')

  axs[0,1].set_title('Original Image')
  axs[0,1].imshow(img)
  axs[0,1].axis('off') 

  axs[1,0].set_title('IG Attribution Mask')
  axs[1,0].imshow(attribution_mask, cmap=cmap)
  axs[1,0].axis('off')  

  axs[1,1].set_title('Original + IG Attribution Mask Overlay')
  axs[1,1].imshow(attribution_mask, cmap=cmap)
  axs[1,1].imshow(img, alpha=overlay_alpha)
  axs[1,1].axis('off')

  plt.tight_layout()

  return fig

name_baseline_tensors = {
    'Baseline Image: Black': tf.zeros(shape=(224,224,3)),
    'Baseline Image: Random': tf.random.uniform(shape=(224,224,3), minval=0.0, maxval=1.0),
    'Baseline Image: White': tf.ones(shape=(224,224,3)),
}