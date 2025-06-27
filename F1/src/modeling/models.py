import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, TensorDataset

class LSTMPredictor(pl.LightningModule):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=2, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        lstm_out, _ = self.lstm(x)
        # Use only the last output for prediction
        out = self.fc(lstm_out[:, -1, :])
        return out
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, _ = batch
        logits = self(x)
        probs = F.softmax(logits, dim=1)
        return probs
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class GRUPredictor(pl.LightningModule):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=2, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        gru_out, _ = self.gru(x)
        # Use only the last output for prediction
        out = self.fc(gru_out[:, -1, :])
        return out
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, _ = batch
        logits = self(x)
        probs = F.softmax(logits, dim=1)
        return probs
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class CNNPredictor(pl.LightningModule):
    def __init__(self, input_size, seq_length, output_size=2, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        # For 1D CNN, we need to transpose the input
        # Input shape: (batch_size, seq_length, input_size)
        # After transpose: (batch_size, input_size, seq_length)
        
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the size after convolutions and pooling
        self.fc_input_size = 128 * (seq_length // 2 // 2)  # Two pooling layers
        
        self.fc1 = nn.Linear(self.fc_input_size, 64)
        self.fc2 = nn.Linear(64, output_size)
        
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        # Transpose for 1D CNN
        x = x.transpose(1, 2)  # Now: (batch_size, input_size, seq_length)
        
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, _ = batch
        logits = self(x)
        probs = F.softmax(logits, dim=1)
        return probs
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class MLPPredictor(pl.LightningModule):
    def __init__(self, input_size, seq_length, hidden_size=128, output_size=2, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        # Flatten the input
        self.flatten_size = input_size * seq_length
        
        self.fc1 = nn.Linear(self.flatten_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        # Flatten
        x = x.reshape(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, _ = batch
        logits = self(x)
        probs = F.softmax(logits, dim=1)
        return probs
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=64):
    """
    Create PyTorch DataLoaders from numpy arrays.
    """
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader
