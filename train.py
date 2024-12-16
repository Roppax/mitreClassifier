# train.py
from model import mitreClassifier
import torch
import os

if not os.path.exists('train_data.pt') or not os.path.exists('val_data.pt'):
    raise FileNotFoundError("Required data files are missing!")

train_data = torch.load('train_data.pt')
val_data = torch.load('val_data.pt')

X_train, y_train = train_data['input_ids'], train_data['labels']
X_val, y_val = val_data['input_ids'], val_data['labels']

model = mitreClassifier(num_labels=len(set(y_train)))  # Adjust based on dataset

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

print("Model loaded and ready for training.")
