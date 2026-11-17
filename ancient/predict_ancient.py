import os, glob, pickle, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
 
DEFAULT_TRAIN_PKL   = 'data/combined_training.pkl'
DEFAULT_ANCIENT_DIR = 'data/ancient'
DEFAULT_OUT_DIR     = 'predictions'
DEFAULT_TABICL_SRC  = ''   # path to tabicl-main/src if TabICL is not pip-installed

DATASETS = {'newdata1': 'newdata1', 'newdata2': 'newdata2', 'newdata3': 'newdata3'}
 
FEATURE_SLICE = slice(40, 80)   # ESM-2 only
FEATURE_DIM   = 40
MAX_SEQ_LEN   = 200
VAL_FRAC      = 0.10
SEED          = 42

class APINModel(nn.Module):
    def __init__(self, input_dim, num_filters=64, filter_range=14, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.convs = nn.ModuleList([
            nn.Conv1d(input_dim, num_filters, kernel_size=(i + 1) * 2, padding='same')
            for i in range(filter_range)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_filters) for _ in range(filter_range)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(num_filters, num_filters // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(num_filters // 2, 2))
 
    def forward(self, x):                 # x: [B, L, D]
        x = x.transpose(1, 2)             # [B, D, L]
        outs = []
        for conv, bn in zip(self.convs, self.bns):
            o = F.relu(bn(conv(x)))
            outs.append(F.max_pool1d(o, o.size(2)).squeeze(2))
        g, _ = torch.max(torch.stack(outs, dim=1), dim=1)
        return self.classifier(self.dropout(g))
 
 
class EBiLSTMModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.bilstm1 = nn.LSTM(input_dim, 128, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.3)
        self.bilstm2 = nn.LSTM(256, 64, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(0.3)
        self.bilstm3 = nn.LSTM(128, 32, batch_first=True, bidirectional=True)
        self.dropout3 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.dropout_fc = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 2)
 
    def forward(self, x, lengths):        # x: [B, Lmax_batch, D]
        B = x.size(0)
        total = x.size(1)
        lengths_sorted, sort_idx = lengths.sort(descending=True)
        x_sorted = x[sort_idx]
        lc = lengths_sorted.clamp(min=1).cpu()
 
        p = pack_padded_sequence(x_sorted, lc, batch_first=True, enforce_sorted=True)
        o, _ = self.bilstm1(p); o, _ = pad_packed_sequence(o, batch_first=True, total_length=total); o = self.dropout1(o)
        p = pack_padded_sequence(o, lc, batch_first=True, enforce_sorted=True)
        o, _ = self.bilstm2(p); o, _ = pad_packed_sequence(o, batch_first=True, total_length=total); o = self.dropout2(o)
        p = pack_padded_sequence(o, lc, batch_first=True, enforce_sorted=True)
        o, _ = self.bilstm3(p); o, _ = pad_packed_sequence(o, batch_first=True, total_length=total); o = self.dropout3(o)
 
        idx = (lc - 1).long()
        fwd = o[torch.arange(B), idx, :32]
        bwd = o[:, 0, 32:]
        h = torch.cat([fwd, bwd], dim=1)
        out = self.fc2(self.dropout_fc(self.relu(self.fc1(h))))
 
        unsort = torch.empty_like(sort_idx)
        unsort[sort_idx] = torch.arange(B, device=sort_idx.device)
        return out[unsort]

class PepDataset(Dataset):
    def __init__(self, feats, labels=None):
        self.feats = feats
        self.labels = labels
    def __len__(self):
        return len(self.feats)
    def __getitem__(self, i):
        a = self.feats[i]
        if a.shape[0] > MAX_SEQ_LEN:
            a = a[:MAX_SEQ_LEN]
        y = -1 if self.labels is None else int(self.labels[i])
        return torch.from_numpy(np.ascontiguousarray(a)), a.shape[0], y
 
 
def collate(batch):
    arrs, lens, ys = zip(*batch)
    Lmax = max(lens)
    X = torch.zeros(len(arrs), Lmax, FEATURE_DIM, dtype=torch.float32)
    for i, a in enumerate(arrs):
        X[i, :a.shape[0]] = a
    return X, torch.tensor(lens, dtype=torch.long), torch.tensor(ys, dtype=torch.long)
 

def load_training(train_pkl):
    with open(train_pkl, 'rb') as f:
        d = pickle.load(f)
    feats = [a[:, FEATURE_SLICE].astype(np.float32) for a in d['features']]
    return feats, np.asarray(d['labels'])
 

def load_ancient(ancient_dir, folder):
    files = sorted(glob.glob(os.path.join(ancient_dir, folder, 'features', '*.npz')))
    seqs, feats = [], []
    for fp in files:
        d = np.load(fp)
        x = d['features'] if 'features' in d.files else d[d.files[0]]
        feats.append(np.asarray(x, dtype=np.float32)[:, FEATURE_SLICE])
        seqs.append(os.path.splitext(os.path.basename(fp))[0])
    return seqs, feats
 

def mean_pool(feats):
    return np.vstack([a.mean(axis=0) for a in feats]).astype(np.float32)



def train_torch(method, tr_feats, tr_y, va_feats, va_y, device):
    if method == 'apin':
        model = APINModel(FEATURE_DIM).to(device)
        opt = optim.RMSprop(model.parameters(), lr=0.0002); epochs, patience = 50, 20
    else:
        model = EBiLSTMModel(FEATURE_DIM).to(device)
        opt = optim.Adam(model.parameters(), lr=0.001); epochs, patience = 500, 20
    crit = nn.CrossEntropyLoss()
 
    tr = DataLoader(PepDataset(tr_feats, tr_y), batch_size=32, shuffle=True,  collate_fn=collate)
    va = DataLoader(PepDataset(va_feats, va_y), batch_size=64, shuffle=False, collate_fn=collate)
 
    best_auc, best_state, bad = 0.0, None, 0
    for ep in range(epochs):
        model.train()
        for X, lens, y in tr:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            out = model(X) if method == 'apin' else model(X, lens)
            crit(out, y).backward(); opt.step()
 
        model.eval(); probs, ys = [], []
        with torch.no_grad():
            for X, lens, y in va:
                X = X.to(device)
                out = model(X) if method == 'apin' else model(X, lens)
                probs.extend(F.softmax(out, 1)[:, 1].cpu().numpy()); ys.extend(y.numpy())
        auc = roc_auc_score(ys, probs) if len(set(ys)) > 1 else 0.0
        if auc > best_auc:
            best_auc, best_state, bad = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if ep % 5 == 0:
            acc = accuracy_score(ys, (np.array(probs) >= .5).astype(int))
            print(f"  epoch {ep:3d}  val AUC {auc:.4f}  val ACC {acc:.4f}  (best {best_auc:.4f})", flush=True)
        if bad >= patience:
            print(f"  early stop at epoch {ep}"); break
    if best_state: model.load_state_dict(best_state)
    print(f"  BEST VAL AUC: {best_auc:.4f}")
    return model


def predict_torch(model, method, feats, device):
    loader = DataLoader(PepDataset(feats), batch_size=256, shuffle=False, collate_fn=collate)
    model.eval(); out_probs = []; done = 0
    with torch.no_grad():
        for X, lens, _ in loader:
            X = X.to(device)
            out = model(X) if method == 'apin' else model(X, lens)
            out_probs.extend(F.softmax(out, 1)[:, 1].cpu().numpy())
            done += X.size(0)
            if done % (256 * 50) < 256:
                print(f"    predicted {done}/{len(feats)}", flush=True)
    return np.array(out_probs)


def save(out_dir, tag, method, seqs, probs):
    df = pd.DataFrame({'sequence': seqs, 'prob': probs, 'pred': (probs >= 0.5).astype(int)})
    out = os.path.join(out_dir, f'{method}_{tag}.csv')
    df.to_csv(out, index=False)
    n = int(df['pred'].sum())
    print(f"  -> {out}  |  {n}/{len(df)} AMP ({100*n/len(df):.2f}%)", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Score ancient peptides with a benchmarked AMP model (ESM-2 features).")
    ap.add_argument('--method', required=True, choices=['apin', 'ebilstm', 'tabicl'])
    ap.add_argument('--train-pkl', default=DEFAULT_TRAIN_PKL,
                    help="pickle with keys 'features' (list of [L,258] arrays) and 'labels'")
    ap.add_argument('--ancient-dir', default=DEFAULT_ANCIENT_DIR,
                    help="directory holding one subfolder per ancient dataset, each with a features/ subdir of .npz files")
    ap.add_argument('--out-dir', default=DEFAULT_OUT_DIR, help="output directory for prediction CSVs")
    ap.add_argument('--tabicl-src', default=DEFAULT_TABICL_SRC,
                    help="path to tabicl-main/src (only if TabICL is not importable on the path)")
    args = ap.parse_args()
    method = args.method
 
    np.random.seed(SEED); torch.manual_seed(SEED)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== {method.upper()} | ESM-2 only (cols 40:80, dim {FEATURE_DIM}) | device {device} ===", flush=True)
 
    feats, y = load_training(args.train_pkl)
    print(f"Loaded {len(feats)} training peptides (pos={int((y==1).sum())}, neg={int((y==0).sum())})")
    idx_tr, idx_va = train_test_split(np.arange(len(y)), test_size=VAL_FRAC,
                                      stratify=y, random_state=SEED)
    tr_feats = [feats[i] for i in idx_tr]; tr_y = y[idx_tr]
    va_feats = [feats[i] for i in idx_va]; va_y = y[idx_va]
    print(f"  train={len(tr_feats)}  val={len(va_feats)} (stratified)")
 
    if method == 'tabicl':
        if args.tabicl_src:
            import sys
            if args.tabicl_src not in sys.path: sys.path.insert(0, args.tabicl_src)
        from tabicl import TabICLClassifier
        clf = TabICLClassifier(
            n_estimators=32, norm_methods=['none', 'power'], feat_shuffle_method='latin',
            class_shift=True, outlier_threshold=4.0, softmax_temperature=0.9,
            average_logits=True, use_hierarchical=True, use_amp=False, batch_size=8,
            device='cpu', random_state=SEED, verbose=False)
        Xtr = mean_pool(tr_feats)
        print("Fitting TabICL (in-context)...", flush=True)
        clf.fit(Xtr, tr_y)
        va_prob = clf.predict_proba(mean_pool(va_feats))[:, 1]
        print(f"  VAL AUC: {roc_auc_score(va_y, va_prob):.4f}")
        CHUNK = 5000
        for tag, folder in DATASETS.items():
            print(f"Predicting {tag}...", flush=True)
            seqs, af = load_ancient(args.ancient_dir, folder)
            Xanc = mean_pool(af)
            probs = np.empty(len(Xanc), dtype=np.float32)
            for i in range(0, len(Xanc), CHUNK):
                probs[i:i+CHUNK] = clf.predict_proba(Xanc[i:i+CHUNK])[:, 1]
                print(f"    {min(i+CHUNK, len(Xanc))}/{len(Xanc)}", flush=True)
            save(args.out_dir, tag, method, seqs, probs)
    else:
        model = train_torch(method, tr_feats, tr_y, va_feats, va_y, device)
        for tag, folder in DATASETS.items():
            print(f"Predicting {tag}...", flush=True)
            seqs, af = load_ancient(args.ancient_dir, folder)
            probs = predict_torch(model, method, af, device)
            save(args.out_dir, tag, method, seqs, probs)
 
 
if __name__ == '__main__':
    main()