import pandas as pd
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import torch

# load clean dataset with LogText column
df_cleaned = pd.read_csv(r"data\evtx_data_with_logtext(1).csv")

# initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# tokenize 'Logtext' column
inputs = tokenizer(list(df_cleaned['LogText']), padding=True, truncation=True, return_tensors='pt')

# prepare labels based on 'EVTX_Tactic'
labels = df_cleaned['EVTX_Tactic'].values

# split data into training 80%, validation 10% and test 10% sets
X_train, X_temp, y_train, y_temp = train_test_split(inputs['input_ids'], labels, test_size=0.2, random_state=42)

# split remaining 20% into validation (50% of 20%, so 10% total) and test (50% of 20%, so 10% total)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# optionally, save tokenized inputs and labels for later use
torch.save({'input_ids': X_train, 'labels': y_train}, 'train_data.pt')
torch.save({'input_ids': X_val, 'labels': y_val}, 'val_data.pt')
torch.save({'input_ids': X_test, 'labels': y_test}, 'test_data.pt')


# load tokenized datasets
train_data = torch.load('train_data.pt')
val_data = torch.load('val_data.pt')
test_data = torch.load('test_data.pt')


# extract input_ids and labels for each set
X_train, y_train = train_data['input_ids'], train_data['labels']
X_val, y_val = val_data['input_ids'], val_data['labels']
X_test, y_test = test_data['input_ids'], test_data['labels']


# check shapes of loaded data
print("Training data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)
print("Test data shape:", X_test.shape)




