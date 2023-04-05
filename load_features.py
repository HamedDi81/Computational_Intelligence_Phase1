import numpy as np
from typing import Tuple

def load_data(train_data: bool =True) -> Tuple[np.ndarray, np.ndarray]:
  """
  loads train/test features with image labels.  
  """
  if train_data:
    data = np.load(f'train_data.npz')
  else:
    data = np.load(f'test_data.npz')

  features = data['features']
  img_labels = data['img_labels']
  
  return features, img_labels

def load_data_with_domain_label() -> Tuple[np.ndarray, np.ndarray]:
  """
  loads portion of training features with domain label
  """
  data = np.load(f'train_data_w_label.npz')
  train_features = data['features']
  domain_labels = data['domain_labels']
  
  return train_features, domain_labels

# Train Data with image labels
Train_features, image_labels = load_data(True)
print(Train_features.shape)
print(image_labels.shape)
print(f'Number of training samples: {Train_features.shape[0]}')

# Test Data with image labels
test_features, image_labels = load_data(False)
print(test_features.shape)
print(image_labels.shape)

# 5% of train data with domain label
train_features, domain_labels = load_data_with_domain_label()
print(train_features.shape)
print(domain_labels.shape)