# Import necessary libraries
import pandas as pd
import re
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, Dataset

# Load and preprocess the dataset
true_df = pd.read_csv('/Users/srishmaulik/Desktop/fake_news_detector/True.csv')
fake_df = pd.read_csv('/Users/srishmaulik/Desktop/fake_news_detector/Fake.csv')

# Add a label column
true_df['label'] = 0  # Real news
fake_df['label'] = 1  # Fake news

# Combine the datasets
df = pd.concat([true_df, fake_df])

# Clean the text
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\W', ' ', text)  # Remove non-words
    text = text.lower()  # Convert to lowercase
    return text

df['text'] = df['text'].apply(clean_text)

# Split the dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'], df['label'], test_size=0.2)

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the texts
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=512)

# Define a Dataset class
class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create dataset objects
train_dataset = NewsDataset(train_encodings, train_labels)
val_dataset = NewsDataset(val_encodings, val_labels)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Load BERT model with a sequence classification head
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Move model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.train()

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Train the model
for epoch in range(3):  # Number of epochs
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1} completed')

# Save the model
torch.save(model.state_dict(), 'bert_fake_news_detector.pth')

# Evaluate the model
model.eval()
val_preds = []
val_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        val_preds.extend(preds.cpu().numpy())
        val_labels.extend(labels.cpu().numpy())

# Calculate accuracy
accuracy = accuracy_score(val_labels, val_preds)
print(f'Validation Accuracy: {accuracy}')

# Print classification report
print(classification_report(val_labels, val_preds, target_names=['Real', 'Fake']))
