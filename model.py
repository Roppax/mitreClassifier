# model.py
import torch
import torch.nn as nn
from transformers import BertModel
from sklearn.preprocessing import LabelEncoder  # import LabelEncoder for encoding labels

# define the model class with BERT backbone and a classification head
class mitreClassifier(nn.Module):
    def __init__(self, num_labels):
        super(mitreClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')  # load the pre-trained BERT model
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)  # add a classifier layer for our task

    def forward(self, input_ids, attention_mask=None):
        # get the output from BERT (including the [CLS] token)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # get the [CLS] token (first token of the sequence)
        logits = self.classifier(cls_output)  # pass through the classifier to get final logits
        return logits

# load training, validation, and test data from the preprocessed .pt files
train_data = torch.load('train_data.pt')
val_data = torch.load('val_data.pt')
test_data = torch.load('test_data.pt')

# extract input features (X) and labels (y) from the loaded datasets
X_train, y_train = train_data['input_ids'], train_data['labels']
X_val, y_val = val_data['input_ids'], val_data['labels']
X_test, y_test = test_data['input_ids'], test_data['labels']

# initialize LabelEncoder to convert string labels into integers
label_encoder = LabelEncoder()

# fit the encoder on y_train and then transform y_train, y_val, and y_test labels
y_train_encoded = label_encoder.fit_transform(y_train)  # encode training labels
y_val_encoded = label_encoder.transform(y_val)  # encode validation labels
y_test_encoded = label_encoder.transform(y_test)  # encode test labels to ensure consistency

# create TensorDatasets for training, validation, and test datasets with the correct label type
train_dataset = torch.utils.data.TensorDataset(X_train, torch.tensor(y_train_encoded, dtype=torch.long))
val_dataset = torch.utils.data.TensorDataset(X_val, torch.tensor(y_val_encoded, dtype=torch.long))
test_dataset = torch.utils.data.TensorDataset(X_test, torch.tensor(y_test_encoded, dtype=torch.long))

# initialize the model with the number of unique labels
model = mitreClassifier(num_labels=len(label_encoder.classes_))  # set num_labels dynamically based on the label encoding

# set up the optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)  # use AdamW optimizer for fine-tuning BERT
loss_fn = nn.CrossEntropyLoss()  # use cross entropy loss for multi-class classification

# move the model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# create DataLoader for the training, validation, and test datasets
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)  # training data in mini-batches
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=16)  # validation data in mini-batches
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16)  # test data in mini-batches

# training loop
epochs = 3  # define the number of training epochs
for epoch in range(epochs):
    model.train()  # set the model to training mode
    total_loss = 0  # initialize total loss for the epoch
    for batch in train_dataloader:
        input_ids, labels = [b.to(device) for b in batch]  # move data to GPU if available
        optimizer.zero_grad()  # zero the gradients before backpropagation
        outputs = model(input_ids)  # perform a forward pass
        loss = loss_fn(outputs, labels)  # calculate the loss
        loss.backward()  # compute gradients via backpropagation
        optimizer.step()  # update model weights using the optimizer
        total_loss += loss.item()  # accumulate the loss for this batch

    # print the average loss for the epoch
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_dataloader)}")

    # validation step after each epoch
    model.eval()  # set the model to evaluation mode
    total_val_loss = 0  # initialize validation loss
    correct_predictions = 0  # initialize correct prediction count for validation accuracy
    with torch.no_grad():  # turn off gradient calculation to speed up validation
        for batch in val_dataloader:
            input_ids, labels = [b.to(device) for b in batch]  # move data to GPU if available
            outputs = model(input_ids)  # perform a forward pass
            val_loss = loss_fn(outputs, labels)  # calculate the validation loss
            total_val_loss += val_loss.item()  # accumulate validation loss
            predicted_classes = torch.argmax(outputs, dim=1)  # get predicted classes (labels)
            correct_predictions += (predicted_classes == labels).sum().item()  # count correct predictions

    # print validation loss and accuracy
    print(f"Validation Loss: {total_val_loss / len(val_dataloader)}")
    print(f"Validation Accuracy: {correct_predictions / len(X_val)}")

# test evaluation after training
model.eval()  # set the model to evaluation mode
total_test_loss = 0  # initialize test loss
correct_predictions = 0  # initialize correct prediction count for test accuracy
with torch.no_grad():  # turn off gradient calculation to speed up test evaluation
    for batch in test_dataloader:
        input_ids, labels = [b.to(device) for b in batch]  # move data to GPU if available
        outputs = model(input_ids)  # perform a forward pass
        test_loss = loss_fn(outputs, labels)  # calculate the test loss
        total_test_loss += test_loss.item()  # accumulate test loss
        predicted_classes = torch.argmax(outputs, dim=1)  # get predicted classes (labels)
        correct_predictions += (predicted_classes == labels).sum().item()  # count correct predictions

# print test loss and accuracy
test_accuracy = correct_predictions / len(X_test)  # calculate test accuracy
print(f"Test Loss: {total_test_loss / len(test_dataloader)}")
print(f"Test Accuracy: {test_accuracy}")
