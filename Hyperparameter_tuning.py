import torch 
import torch.nn as nn
import torch.optim as optim
from Data_loader import *
import optuna
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim,hidden_dim,dropout_rate):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.batchnorm = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, inputs):
        embeds = self.embeddings(inputs)     # [batch, context_size, embed_dim]
        mean_embeds = embeds.mean(dim=1)     # [batch, embed_dim]
        out = self.linear1(mean_embeds)  
        out = self.batchnorm(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return (out)
    
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_percentage_error

def objective(trial):
    # --- Hyperparameter search space ---
    hidden_dim = trial.suggest_int("hidden_dim", 50, 500, step=50)
    epochs = trial.suggest_int("epochs", 10, 150, step=10)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])

    vocab_size = len(tokenized_text)
    embed_dim = 100

    # --- Model initialization ---
    model = CBOW(vocab_size, embed_dim, hidden_dim, dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()

    # --- Optimizer selection ---
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # --- Training loop ---
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        trial.report(total_loss / len(train_loader), epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # --- Evaluation ---
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(Y_batch.cpu().numpy())

    # --- Metric ---
    mape = mean_absolute_percentage_error(all_targets, all_preds)
    return mape


study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=50)
