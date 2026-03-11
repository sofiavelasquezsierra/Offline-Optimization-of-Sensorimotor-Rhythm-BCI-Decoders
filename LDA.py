import os
import glob
import numpy as np
from scipy.io import loadmat
from scipy import signal
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from CSP import csp_fit, csp_transform

DATA_ROOT = './'  
TASK_TO_ANALYZE = 'UD'  # UD or LR

GELLED_ELECTRODES = [
    'F3','F4','FC5','FC3','FC1','FCz','FC2','FC4','FC6',
    'T7','C5','C3','C1','Cz','C2','C4','C6','T8',
    'CP5','CP3','CP1','CPz','CP2','CP4','CP6','P3','P4'
]

class CSPWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, m_pairs=3):
        self.m_pairs = m_pairs
        self.W = None
    
    def fit(self, X, y):
        X1 = X[y == 1]
        X2 = X[y == 2]
        self.W = csp_fit(X1, X2, m_pairs=self.m_pairs)
        return self
    
    def transform(self, X):
        return csp_transform(self.W, X)

# DATA LOADING
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
        pass 

    eeg_data = runData.allData[keep_indices, :]
    fs = int(runData.fs)
    trial_starts = runData.trialStart
    targets = runData.target
    outcomes = runData.outcome

    # Epoching (0s to 4s) - Feedback Period
    n_pre = int(0.0 * fs)
    n_post = int(4.0 * fs)
    
    trials = []
    labels = []
    
    for i, start_idx in enumerate(trial_starts):
        # Skip Aborts
        if outcomes[i] == 0:
            continue
            
        t_start = int(start_idx - n_pre)
        t_end = int(start_idx + n_post)
        
        if t_start < 0 or t_end > eeg_data.shape[1]:
            continue
            
        epoch = eeg_data[:, t_start:t_end]
        trials.append(epoch)
        labels.append(targets[i])
        
    return np.array(trials), np.array(labels)

if __name__ == "__main__":
    all_files = glob.glob(os.path.join(DATA_ROOT, '**', '*.mat'), recursive=True)
    task_files = [f for f in all_files if TASK_TO_ANALYZE in os.path.basename(f)]
    
    subjects = {}
    for f in task_files:
        subj_id = os.path.basename(os.path.dirname(f))
        if subj_id not in subjects:
            subjects[subj_id] = []
        subjects[subj_id].append(f)
    
    for subj, files in subjects.items():
        all_trials = []
        all_labels = []
        
        for f in files:
            t, l = load_and_epoch_file(f, GELLED_ELECTRODES)
            if t is not None and len(t) > 0:
                all_trials.append(t)
                all_labels.append(l)
        
        if not all_trials:
            continue

        X = np.concatenate(all_trials, axis=0)
        y = np.concatenate(all_labels, axis=0)
        
        # Preprocessing
        X = X - X.mean(axis=1, keepdims=True) # CAR
        
        fs = 1000 
        b, a = signal.butter(4, [8, 13], btype='bandpass', fs=fs) # Mu Band
        X = signal.filtfilt(b, a, X, axis=-1)
        
        target_fs = 100
        new_len = int(X.shape[-1] * target_fs / fs)
        X = signal.resample(X, new_len, axis=-1)

        # Pipeline
        clf = Pipeline([
            ('csp', CSPWrapper(m_pairs=3)),
            ('lda', LinearDiscriminantAnalysis())
        ])
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv)
        
        print(f"Subject: {subj} | Trials: {len(y)} | Accuracy: {scores.mean()*100:.2f}% (+/- {scores.std()*100:.2f})")