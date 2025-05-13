import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Union, Tuple

class EnsembleModel:
    """
    Ensemble models that combines predictions from multiple models.
    """
    def __init__(self, models: List[nn.Module], ensemble_method: str = 'average', weights: List[float] = None):
        """
        Initialize the ensemble models.
        
        Args:
            models: List of trained models to ensemble
            ensemble_method: Method to combine predictions ('average', 'weighted_average', or 'voting')
            weights: Weights for each models (only used if ensemble_method is 'weighted_average')
        """
        self.models = models
        self.ensemble_method = ensemble_method
        
        # If weights are not provided, use equal weights
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            # Normalize weights to sum to 1
            total = sum(weights)
            self.weights = [w / total for w in weights]
            
        # Move models to evaluation mode
        for model in self.models:
            model.eval()
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions using the ensemble.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            
        Returns:
            Tensor of shape (batch_size, num_classes) with class probabilities
        """
        with torch.no_grad():
            predictions = [model(x) for model in self.models]
            
            # Apply softmax to get probabilities
            probabilities = [F.softmax(pred, dim=1) for pred in predictions]
            
            if self.ensemble_method == 'average':
                # Simple average of probabilities
                ensemble_pred = torch.mean(torch.stack(probabilities), dim=0)
                
            elif self.ensemble_method == 'weighted_average':
                # Weighted average of probabilities
                weighted_probs = [prob * weight for prob, weight in zip(probabilities, self.weights)]
                ensemble_pred = torch.sum(torch.stack(weighted_probs), dim=0)
                
            elif self.ensemble_method == 'voting':
                # Hard voting (majority vote)
                # Get class predictions from each models
                class_preds = [torch.argmax(prob, dim=1) for prob in probabilities]
                stacked_preds = torch.stack(class_preds)
                
                # Count votes for each class
                batch_size = x.size(0)
                num_classes = probabilities[0].size(1)
                ensemble_pred = torch.zeros((batch_size, num_classes), device=x.device)
                
                for i in range(batch_size):
                    # Count votes for each class in this sample
                    unique_classes, counts = torch.unique(stacked_preds[:, i], return_counts=True)
                    for cls, count in zip(unique_classes, counts):
                        ensemble_pred[i, cls] = count / len(self.models)
            else:
                raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
                
            return ensemble_pred
    
    def predict_classes(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels using the ensemble.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            
        Returns:
            Tensor of shape (batch_size,) with class predictions
        """
        probs = self.predict(x)
        return torch.argmax(probs, dim=1)
    
    def evaluate(self, dataloader) -> Dict[str, float]:
        """
        Evaluate the ensemble models on a dataset.
        
        Args:
            dataloader: PyTorch DataLoader containing evaluation data
            
        Returns:
            Dictionary with evaluation metrics
        """
        correct = 0
        total = 0
        
        for x, y in dataloader:
            batch_size = x.size(0)
            total += batch_size
            
            preds = self.predict_classes(x)
            correct += (preds == y).sum().item()
        
        accuracy = correct / total
        return {'accuracy': accuracy}
    
    def get_model_weights(self) -> List[float]:
        """
        Get the weights assigned to each models.
        
        Returns:
            List of weights for each models
        """
        return self.weights
    
    def set_model_weights(self, weights: List[float]) -> None:
        """
        Set the weights for each models.
        
        Args:
            weights: List of weights for each models
        """
        # Normalize weights to sum to 1
        total = sum(weights)
        self.weights = [w / total for w in weights]
    
    def optimize_weights(self, val_dataloader, num_iterations: int = 100, learning_rate: float = 0.01) -> None:
        """
        Optimize the weights of the ensemble using gradient descent on validation data.
        
        Args:
            val_dataloader: PyTorch DataLoader containing validation data
            num_iterations: Number of optimization iterations
            learning_rate: Learning rate for gradient descent
        """
        # Initialize weights
        weights = torch.tensor(self.weights, requires_grad=True)
        
        # Collect all models predictions on validation data
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in val_dataloader:
                model_preds = []
                for model in self.models:
                    logits = model(x)
                    probs = F.softmax(logits, dim=1)
                    model_preds.append(probs)
                
                all_preds.append(model_preds)
                all_targets.append(y)
        
        # Optimize weights using gradient descent
        for iteration in range(num_iterations):
            total_loss = 0
            
            for batch_idx, (batch_preds, targets) in enumerate(zip(all_preds, all_targets)):
                # Normalize weights using softmax
                norm_weights = F.softmax(weights, dim=0)
                
                # Compute weighted ensemble prediction
                ensemble_pred = torch.zeros_like(batch_preds[0])
                for i, pred in enumerate(batch_preds):
                    ensemble_pred += pred * norm_weights[i]
                
                # Compute cross-entropy loss
                loss = F.cross_entropy(ensemble_pred, targets)
                total_loss += loss
                
                # Backward pass
                loss.backward()
            
            # Update weights
            with torch.no_grad():
                weights -= learning_rate * weights.grad
                weights.grad.zero_()
            
            # Print progress
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{num_iterations}, Loss: {total_loss.item():.4f}")
        
        # Update ensemble weights
        final_weights = F.softmax(weights, dim=0).tolist()
        self.set_model_weights(final_weights)
        print(f"Optimized weights: {self.weights}")
