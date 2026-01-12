"""
Helper Script: Save Your Model Predictions
===========================================

This script helps you save your trained model predictions in the correct format
for the evaluation pipeline.

Example Usage:
--------------
from save_predictions import save_model_predictions

# After training your model and getting predictions on test set:
save_model_predictions(
    predictions=your_predictions_array,
    model_name='LSTM',
    fd_number=1
)
"""

import numpy as np
import pickle
import pandas as pd
import os


def save_model_predictions(predictions: np.ndarray, 
                          model_name: str, 
                          fd_number: int,
                          output_dir: str = './predictions'):
    """
    Save model predictions in the correct format.
    
    Args:
        predictions: NumPy array of RUL predictions
        model_name: Name of the model (e.g., 'SVR', 'LSTM', 'TCN', 'Transformer')
        fd_number: FD dataset number (1-4)
        output_dir: Directory to save predictions
    """
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f'{output_dir}/{model_name}_FD00{fd_number}_predictions.npy'
    np.save(filename, predictions)
    
    print(f"✓ Saved {model_name} predictions to: {filename}")
    print(f"  Shape: {predictions.shape}")
    print(f"  Min: {predictions.min():.2f}, Max: {predictions.max():.2f}, Mean: {predictions.mean():.2f}")


def load_predictions_from_pytorch_model(model_path: str, 
                                       test_loader,
                                       device: str = 'cpu'):
    """
    Load predictions from a saved PyTorch model.
    
    Args:
        model_path: Path to saved model (.pth file)
        test_loader: PyTorch DataLoader with test data
        device: 'cpu' or 'cuda'
    
    Returns:
        predictions: NumPy array of predictions
        true_values: NumPy array of true RUL values
    """
    import torch
    
    # Load model (you'll need to replace this with your actual model class)
    # model = YourModelClass()
    # model.load_state_dict(torch.load(model_path, map_location=device))
    # model.eval()
    
    predictions = []
    true_values = []
    
    # Get predictions
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)
            
            predictions.extend(y_pred.cpu().numpy().flatten())
            true_values.extend(y_batch.numpy().flatten())
    
    return np.array(predictions), np.array(true_values)


def example_pytorch_evaluation():
    """
    Example: How to evaluate a PyTorch model and save predictions.
    
    REPLACE THIS WITH YOUR ACTUAL MODEL CODE
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    
    # === EXAMPLE MODEL ARCHITECTURE (REPLACE WITH YOUR ACTUAL MODEL) ===
    class ExampleLSTM(nn.Module):
        def __init__(self, input_dim, hidden_dim=128, num_layers=2):
            super(ExampleLSTM, self).__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                               batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_dim, 1)
        
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.fc(lstm_out[:, -1, :])
    
    # === LOAD YOUR TEST DATA (REPLACE WITH YOUR ACTUAL DATA LOADING) ===
    # Example: Create dummy data
    window_size = 30
    num_features = 14
    n_test_samples = 100
    
    X_test = torch.randn(n_test_samples, window_size, num_features)
    y_test = torch.randn(n_test_samples, 1) * 50 + 50  # RUL values
    
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # === LOAD YOUR TRAINED MODEL ===
    model = ExampleLSTM(input_dim=num_features)
    # model.load_state_dict(torch.load('path/to/your/model.pth'))
    model.eval()
    
    # === GET PREDICTIONS ===
    predictions = []
    true_values = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch)
            predictions.extend(y_pred.numpy().flatten())
            true_values.extend(y_batch.numpy().flatten())
    
    predictions = np.array(predictions)
    true_values = np.array(true_values)
    
    # === SAVE PREDICTIONS ===
    save_model_predictions(predictions, model_name='LSTM', fd_number=1)
    
    print("\n✓ Predictions saved! Now run: python run_evaluation.py")


def example_sklearn_evaluation():
    """
    Example: How to evaluate a scikit-learn model and save predictions.
    """
    from sklearn.svm import SVR
    import pickle
    
    # === LOAD YOUR TRAINED MODEL ===
    # with open('path/to/your/svr_model.pkl', 'rb') as f:
    #     model = pickle.load(f)
    
    # Create example model
    model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    
    # === LOAD YOUR TEST DATA ===
    # X_test, y_test = load_your_test_data()
    
    # Example: Create dummy flattened data (window_size * num_features)
    n_test_samples = 100
    X_test = np.random.randn(n_test_samples, 30 * 14)  # Flattened sequences
    y_test = np.random.exponential(50, n_test_samples)
    
    # Train dummy model
    model.fit(X_test[:80], y_test[:80])
    
    # === GET PREDICTIONS ===
    predictions = model.predict(X_test[80:])
    true_values = y_test[80:]
    
    # === SAVE PREDICTIONS ===
    save_model_predictions(predictions, model_name='SVR', fd_number=1)
    
    print("\n✓ Predictions saved! Now run: python run_evaluation.py")


def verify_predictions_format(predictions_dir: str = './predictions'):
    """
    Verify that all prediction files are in the correct format.
    """
    print("\n" + "="*60)
    print("VERIFYING PREDICTION FILES")
    print("="*60)
    
    required_models = ['SVR', 'LSTM', 'TCN', 'Transformer']
    fd_numbers = [1, 3]  # FD001 and FD003
    
    all_good = True
    
    for fd in fd_numbers:
        print(f"\nFD00{fd}:")
        for model in required_models:
            filename = f'{predictions_dir}/{model}_FD00{fd}_predictions.npy'
            
            if os.path.exists(filename):
                try:
                    preds = np.load(filename)
                    print(f"  ✓ {model}: {preds.shape} - OK")
                except Exception as e:
                    print(f"  ✗ {model}: Error loading - {e}")
                    all_good = False
            else:
                print(f"  ⚠ {model}: File not found")
                all_good = False
    
    print("\n" + "="*60)
    if all_good:
        print("✓ All predictions verified! Ready to run evaluation.")
    else:
        print("⚠ Some predictions are missing or invalid.")
        print("  Use save_model_predictions() to save your predictions.")
    print("="*60)


def quick_start_template():
    """
    Quick start template for your actual code.
    Copy this and fill in with your models.
    """
    template = '''
# === QUICK START TEMPLATE ===
# Copy this code and replace with your actual models

import numpy as np
from save_predictions import save_model_predictions

# 1. Load your trained models
# lstm_model = load_model('path/to/lstm.pth')
# tcn_model = load_model('path/to/tcn.pth')
# transformer_model = load_model('path/to/transformer.pth')
# svr_model = load_model('path/to/svr.pkl')

# 2. Load your test data
# X_test, y_test = load_test_data('FD001')

# 3. Get predictions for each model
# lstm_preds = lstm_model.predict(X_test)
# tcn_preds = tcn_model.predict(X_test)
# transformer_preds = transformer_model.predict(X_test)
# svr_preds = svr_model.predict(X_test)

# 4. Save predictions
# save_model_predictions(lstm_preds, 'LSTM', fd_number=1)
# save_model_predictions(tcn_preds, 'TCN', fd_number=1)
# save_model_predictions(transformer_preds, 'Transformer', fd_number=1)
# save_model_predictions(svr_preds, 'SVR', fd_number=1)

# 5. Run evaluation
# python run_evaluation.py

print("Template ready! Fill in with your actual model code.")
    '''
    
    print(template)
    
    # Save template to file
    with open('quick_start_template.py', 'w') as f:
        f.write(template)
    
    print("\n✓ Template saved to: quick_start_template.py")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("PREDICTION SAVER UTILITY")
    print("="*60)
    
    print("\n📋 Available Functions:")
    print("  1. save_model_predictions() - Save predictions")
    print("  2. verify_predictions_format() - Check saved predictions")
    print("  3. example_pytorch_evaluation() - PyTorch example")
    print("  4. example_sklearn_evaluation() - Sklearn example")
    print("  5. quick_start_template() - Generate starter code")
    
    print("\n" + "="*60)
    print("RUNNING VERIFICATION...")
    print("="*60)
    
    # Check current predictions
    verify_predictions_format()
    
    print("\n\n💡 TO USE THIS UTILITY:")
    print("="*60)
    print("1. Import: from save_predictions import save_model_predictions")
    print("2. Get your model predictions: preds = model.predict(X_test)")
    print("3. Save: save_model_predictions(preds, 'LSTM', fd_number=1)")
    print("4. Run: python run_evaluation.py")
    print("="*60)
    
    # Generate quick start template
    print("\n\nGenerating quick start template...")
    quick_start_template()
