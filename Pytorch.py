import torch
print(torch.__version__)
#Importing the dataset
import pandas as pd
df = pd.read_csv("IMDB Dataset_cleaned_10K.csv")

#Tokenizer and Create Vocabulary
from torchtext.data import get_tokenizer
#Defining the Tokenizer Function
tokenized_text = []
def tokenization(text):
    tokenizer = get_tokenizer("basic_english")
    for i in range (len(df)):
        token = tokenizer(df.iloc[i,0])
        tokenized_text.append(token)
    return tokenized_text

#Deploying Tokenizer
tokenization(df['review'])

#Concatination to a single list or a numpy array
import numpy as np
tokenized_text = np.concatenate(tokenized_text)
tokenized_text = sorted(set(tokenized_text))
#tokenized_text = np.concatenate(tokenized_text)

#Creating a key:value pair for conversion between number and words
word_to_idx = {word : i for i,word in enumerate(tokenized_text)}
idx_to_word = {i : word for i,word in enumerate(tokenized_text)}

#Creating dataset for context and target
rows = []
for i in range(2,len(tokenized_text)-2):
    rows.append({
        "--": tokenized_text[i - 2],
        "-": tokenized_text[i - 1],
        "+": tokenized_text[i + 1],
        "++": tokenized_text[i + 2],
        "Target": tokenized_text[i]
        })
context = pd.DataFrame(rows)
context
#Converting all the words into numbers
def word_to_num(text):
    return (word_to_idx[text])


context = context.applymap(word_to_num)
context

# X and Y extraction
X = context.iloc[:,:4].values
Y = context.iloc[:,-1].values

#Train test split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

#Embedding Part
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#X_train_tensor = torch.from_numpy(X_train)
#X_test_tensor = torch.from_numpy(X_test)
#Y_train_tensor = torch.from_numpy(Y_train)
#Y_test_tensor = torch.from_numpy(Y_test)

#Creating a Dataset and Dataloader class
from torch.utils.data import Dataset,DataLoader
class CustomDataset(Dataset):
    def __init__(self,features,labels):
        self.features = torch.tensor(features,dtype=torch.long,device=device)
        self.labels = torch.tensor(labels,dtype=torch.long,device=device)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self,index):
        return self.features[index],self.labels[index]
    
train_data = CustomDataset(X_train,Y_train)
test_data = CustomDataset(X_test, Y_test)

#Creating train and test loader
train_loader = DataLoader(train_data,batch_size=512,shuffle=True)
test_loader = DataLoader(test_data,batch_size=512,shuffle=False)

#Creating Embedding module
import torch.nn as nn
import torch.optim as optim

'''class CBOW(nn.Module):
    def __init__(self,embedding_dim, vocab_size):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self,inputs):
        embedding = self.embeddings(inputs).mean(1).squeeze(1)
        return self.linear(embedding)
'''

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
    def forward(self, inputs):
        embeds = self.embeddings(inputs)     # [batch, context_size, embed_dim]
        mean_embeds = embeds.mean(dim=1)     # [batch, embed_dim]
        out = self.linear(mean_embeds)       # [batch, vocab_size]
        return out

        
vocab_size = len(tokenized_text)
embed_dim = 100
model = CBOW(vocab_size,embed_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(),0.01)
criterion = nn.CrossEntropyLoss()

print("Training Starts..........")
for epoch in range(100):

  total_epoch_loss = 0

  for batch_features, batch_labels in train_loader:

    # forward pass
    outputs = model(batch_features)

    # calculate loss
    loss = criterion(outputs, batch_labels)

    # back pass
    optimizer.zero_grad()
    loss.backward()

    # update grads
    optimizer.step()

    total_epoch_loss = total_epoch_loss + loss.item()

  avg_loss = total_epoch_loss/len(train_loader)
  print(f'Epoch: {epoch + 1} , Loss: {avg_loss}')         

#Testing Phase
with torch.no_grad():
    for X_batch, Y_batch in test_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        outputs = model(X_batch)                    # [batch, vocab_size]
        predictions = torch.argmax(outputs, dim=1)
Y_batch = Y_batch.detach().cpu().numpy()mbeddings = model.embeddings.weight.data  # shape: [vocab_size, embedding_dim]
embeddings_np = embeddings.cpu().numpy()  # shape: [vocab_size, embedding_dim]
from sklearn.decomposition import PCA
predictions = predictions.detach().cpu().numpy()

from sklearn.metrics import mean_absolute_percentage_error  
mse = mean_absolute_percentage_error(Y_batch, predictions)
print(mse)

word = 'hello'
context_idx = torch.tensor([[word_to_idx[word]] * 4], dtype=torch.long, device=device)
model.eval()
with torch.no_grad():
    output = model(context_idx)
    predicted_idx = torch.argmax(output,dim=1)
    predicted_idx = predicted_idx.detach().cpu()
    print(idx_to_word[predicted_idx.item()])
predicted_idx

context_words = ["movie", "really","awesome"]
context_idx = torch.tensor([[word_to_idx[w] for w in context_words]],
                           dtype=torch.long, device=device)
model.eval()
with torch.no_grad():
    output = model(context_idx)
    predicted_idx = torch.argmax(output, dim=1).item()
    print("Predicted target word:", idx_to_word[predicted_idx])

#Visualization
embeddings = model.embeddings.weight.data  # shape: [vocab_size, embedding_dim]
embeddings_np = embeddings.cpu().numpy()  # shape: [vocab_size, embedding_dim]
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
embeddings_3d = pca.fit_transform(embeddings_np)
words = [idx_to_word[i] for i in range(len(idx_to_word))]

import plotly.express as px
import pandas as pd

# Create a DataFrame
df_plot = pd.DataFrame({
    'x': embeddings_3d[:, 0],
    'y': embeddings_3d[:, 1],
    'z': embeddings_3d[:, 2],
    'word': words
})

# 3D scatter plot
fig = px.scatter_3d(
    df_plot, x='x', y='y', z='z', text='word',
    title='CBOW Word Embeddings in 3D (PCA)'
)

fig.update_traces(marker=dict(size=3), selector=dict(mode='markers+text'))
fig.show()
