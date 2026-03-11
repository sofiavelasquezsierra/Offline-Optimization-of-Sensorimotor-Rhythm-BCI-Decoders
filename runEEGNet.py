import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import signal
import glob
import os

from EEGNet import EEGNet 

DATA_ROOT = './'  
TASK_TO_ANALYZE = 'UD' # UD or LR

GELLED_ELECTRODES = [
    'F3','F4','FC5','FC3','FC1','FCz','FC2','FC4','FC6',
    'T7','C5','C3','C1','Cz','C2','C4','C6','T8',
    'CP5','CP3','CP1','CPz','CP2','CP4','CP6','P3','P4'
]

def load_and_epoch_file(filepath, gelled_names):
    try:
        mat = loadmat(filepath, squeeze_me=True, struct_as_record=False)
        runData = mat['runData']
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None

    # Channel Selection
    raw_labels = [str(l).upper() for l in runData.label]
    target_labels = [g.upper() for g in gelled_names]
    keep_indices = [i for i, lbl in enumerate(raw_labels) if lbl in target_labels]
    
    if len(keep_indices) < len(target_labels):
        return None, None

    eeg_data = runData.allData[keep_indices, :]
    fs = int(runData.fs)
    trial_starts = runData.trialStart
    targets = runData.target
    outcomes = runData.outcome 

    # Epoching (0s to 4s)
    n_pre = int(0.0 * fs) 
    n_post = int(4.0 * fs) 
    
    trials = []
    labels = []
    
    for i, start_idx in enumerate(trial_starts):
        if outcomes[i] == 0: # Skip Abort
            continue
            
        t_start = int(start_idx - n_pre)
        t_end = int(start_idx + n_post)
        
        if t_start < 0 or t_end > eeg_data.shape[1]:
            continue
            
        epoch = eeg_data[:, t_start:t_end]
        trials.append(epoch)
        labels.append(targets[i])
        
    return np.array(trials), np.array(labels)

# TRAINING LOOP
def train_model(model, train_loader, val_loader, epochs=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_acc = 0.0
    best_preds = []
    best_targets = []
    
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        epoch_preds = []
        epoch_targets = []
        
        with torch.no_grad():
            for X_val, y_val in val_loader:
                outputs = model(X_val)
                _, predicted = torch.max(outputs.data, 1)
                total += y_val.size(0)
                correct += (predicted == y_val).sum().item()
                
                # Store predictions for confusion matrix
                epoch_preds.extend(predicted.numpy())
                epoch_targets.extend(y_val.numpy())
        
        acc = 100 * correct / total
        
        # Save the predictions from the best epoch
        if acc > best_acc:
            best_acc = acc
            best_preds = epoch_preds
            best_targets = epoch_targets
            
    return best_acc, best_preds, best_targets

#  MAIN EXECUTION
if __name__ == "__main__":
    # Find & Load Data
    all_files = glob.glob(os.path.join(DATA_ROOT, '**', '*.mat'), recursive=True)
    task_files = [f for f in all_files if TASK_TO_ANALYZE in os.path.basename(f)]
    
    if not task_files: exit()

    all_trials = []
    all_labels = []
    
    for f in task_files:
        t, l = load_and_epoch_file(f, GELLED_ELECTRODES)
        if t is not None and len(t) > 0:
            all_trials.append(t)
            all_labels.append(l)

    X = np.concatenate(all_trials, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    # Preprocessing
    X = X - X.mean(axis=1, keepdims=True) # CAR
    fs = 1000 
    b, a = signal.butter(4, [8, 13], btype='bandpass', fs=fs)
    X = signal.filtfilt(b, a, X, axis=-1)
    
    target_fs = 100
    num_samples_new = int(X.shape[-1] * target_fs / fs)
    X = signal.resample(X, num_samples_new, axis=-1)
    
    # Prepare PyTorch Data
    X_torch = X[:, np.newaxis, :, :] 
    y_torch = y - 1 
    tensor_x = torch.Tensor(X_torch)
    tensor_y = torch.Tensor(y_torch).long()
    
    # 5-Fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Accumulate all predictions from all folds
    all_y_true = []
    all_y_pred = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = tensor_x[train_idx], tensor_x[test_idx]
        y_train, y_test = tensor_y[train_idx], tensor_y[test_idx]
        
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        model = EEGNet(nb_classes=2, Chans=X.shape[1], Samples=X.shape[-1])
        
        acc, preds, targets = train_model(model, train_loader, test_loader, epochs=50)
        
        # Store results for the matrix
        all_y_true.extend(targets)
        all_y_pred.extend(preds)
        
        print(f"Fold {fold+1}: {acc:.2f}%")

    # Generate Confusion Matrix
    cm = confusion_matrix(all_y_true, all_y_pred)
    
    # Plotting
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Up', 'Down'])
    
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    ax.set_title(f'Confusion Matrix: EEGNet (27 Elec)\nAccuracy: {np.mean(np.array(all_y_true) == np.array(all_y_pred))*100:.1f}%')
    
    plt.tight_layout()
    plt.savefig('Figure4_ConfusionMatrix.png', dpi=300)
    plt.show()
    print("Saved as 'Figure4_ConfusionMatrix.png'")