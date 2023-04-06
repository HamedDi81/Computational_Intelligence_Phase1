import numpy as np
from typing import Tuple

def load_data(train_data: bool =True) -> Tuple[np.ndarray, np.ndarray]:
  """
  loads train/test features with image labels.  
  """
  if train_data:
    data = np.load(f'/home/cyrus/features/train_data.npz')
  else:
    data = np.load(f'/home/cyrus/features/test_data.npz')

  features = data['features']
  img_labels = data['img_labels']
  
  return features, img_labels

def load_data_with_domain_label() -> Tuple[np.ndarray, np.ndarray]:
  """
  loads portion of training features with domain label
  """
  data = np.load(f'/home/cyrus/features/train_data_w_label.npz')
  train_features = data['features']
  domain_labels = data['domain_labels']
  
  return train_features, domain_labels
