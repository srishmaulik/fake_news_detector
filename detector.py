import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from pandas import DataFrame as df
# Set up plotting styles
plt.style.use('ggplot')
sns.color_palette("tab10")
sns.set(context='notebook', style='darkgrid', font='sans-serif', font_scale=1, rc=None)
matplotlib.rcParams['figure.figsize'] =[20,8]
matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams['font.family'] = 'sans-serif'

# Load datasets

true_df = pd.read_csv('True.csv')
fake_df = pd.read_csv('Fake.csv')

# Add labels
true_df['label'] = 'REAL'
fake_df['label'] = 'FAKE'

# Combine datasets
df = pd.concat([true_df, fake_df])
# df['label'] = df['label'].astype('category')  # Line 1
# plt.title('The number of news: fake/real')
# plt.show()

# Plot the number of real and fake news
sns.countplot(df.label)  
# Display shape and first few rows
print(df.shape)
print(df.head())

# Extract labels
labels = df.label
print(labels.head())

# Plot the number of real and fake news
sns.countplot(df.label)
plt.title('The number of news: fake/real')
plt.show()

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform train set, transform test set
tfidf_train = tfidf_vectorizer.fit_transform(x_train) 
tfidf_test = tfidf_vectorizer.transform(x_test)

# Initialize PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# Predict on the test set and calculate accuracy
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score*100,2)}%')

# Print confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
print(cm)

# Plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Plot non-normalized confusion matrix
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'], normalize=True,
                      title='Normalized confusion matrix')

plt.show()

# Load and preprocess the dataset
# true_df = pd.read_csv('/Users/srishmaulik/Desktop/fake_news_detector/True.csv')
# fake_df = pd.read_csv('/Users/srishmaulik/Desktop/fake_news_detector/Fake.csv')

# # Add a label column
# true_df['label'] = 0  # Real news
# fake_df['label'] = 1  # Fake news

# # Combine the datasets
# df = pd.concat([true_df, fake_df])

# # Clean the text
# def clean_text(text):
#     text = re.sub(r'http\S+', '', text)  # Remove URLs
#     text = re.sub(r'\W', ' ', text)  # Remove non-words
#     text = text.lower()  # Convert to lowercase
#     return text

# df['text'] = df['text'].apply(clean_text)

# # Split the dataset
# train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'], df['label'], test_size=0.2)

# # Load the BERT tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# # Tokenize the texts
# train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=512)
# val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=512)

# # Define a Dataset class
# class NewsDataset(Dataset):
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels

#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         item['labels'] = torch.tensor(self.labels[idx])
#         return item

#     def __len__(self):
#         return len(self.labels)

# # Create dataset objects
# train_dataset = NewsDataset(train_encodings, train_labels)
# val_dataset = NewsDataset(val_encodings, val_labels)

# # Create data loaders
# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# # Load BERT model with a sequence classification head
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# # Move model to GPU if available
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model.to(device)
# model.train()

# # Define the optimizer
# optimizer = AdamW(model.parameters(), lr=1e-5)

# # Train the model
# for epoch in range(3):  # Number of epochs
#     for batch in train_loader:
#         optimizer.zero_grad()
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)
#         outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()
#     print(f'Epoch {epoch+1} completed')

# # Save the model
# torch.save(model.state_dict(), 'bert_fake_news_detector.pth')

# # Evaluate the model
# model.eval()
# val_preds = []
# val_labels = []

# with torch.no_grad():
#     for batch in val_loader:
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)
#         outputs = model(input_ids, attention_mask=attention_mask)
#         preds = torch.argmax(outputs.logits, dim=1)
#         val_preds.extend(preds.cpu().numpy())
#         val_labels.extend(labels.cpu().numpy())

# # Calculate accuracy
# accuracy = accuracy_score(val_labels, val_preds)
# print(f'Validation Accuracy: {accuracy}')

# # Print classification report
# print(classification_report(val_labels, val_preds, target_names=['Real', 'Fake']))
