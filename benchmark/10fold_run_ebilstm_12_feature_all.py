import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data
from data_processing import load_all_xuamp_data, create_two_stage_random_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef, confusion_matrix


class EBiLSTMModel(nn.Module):
    def __init__(self, input_dim, max_seq_len=200):
        super(EBiLSTMModel, self).__init__()
        
        self.input_dim = input_dim
        self.max_seq_len = max_seq_len
        
        self.bilstm1 = nn.LSTM(
            input_size=input_dim, hidden_size=128,
            batch_first=True, bidirectional=True
        )
        self.dropout1 = nn.Dropout(0.3)
        
        self.bilstm2 = nn.LSTM(
            input_size=256, hidden_size=64,
            batch_first=True, bidirectional=True
        )
        self.dropout2 = nn.Dropout(0.3)
        
        self.bilstm3 = nn.LSTM(
            input_size=128, hidden_size=32,
            batch_first=True, bidirectional=True
        )
        self.dropout3 = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.dropout_fc = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 2)
    
    def forward(self, x, lengths):
        batch_size = x.size(0)
        
        lengths_sorted, sort_idx = lengths.sort(descending=True)
        x_sorted = x[sort_idx]
        
        lengths_clamped = lengths_sorted.clamp(min=1).cpu()
        
        packed = pack_padded_sequence(x_sorted, lengths_clamped, batch_first=True, enforce_sorted=True)
        out1, _ = self.bilstm1(packed)
        out1, _ = pad_packed_sequence(out1, batch_first=True, total_length=self.max_seq_len)
        out1 = self.dropout1(out1)
        
        packed = pack_padded_sequence(out1, lengths_clamped, batch_first=True, enforce_sorted=True)
        out2, _ = self.bilstm2(packed)
        out2, _ = pad_packed_sequence(out2, batch_first=True, total_length=self.max_seq_len)
        out2 = self.dropout2(out2)
        
        packed = pack_padded_sequence(out2, lengths_clamped, batch_first=True, enforce_sorted=True)
        out3, _ = self.bilstm3(packed)
        out3, _ = pad_packed_sequence(out3, batch_first=True, total_length=self.max_seq_len)
        out3 = self.dropout3(out3)
        
        idx = (lengths_clamped - 1).long()
        
        forward_out = out3[torch.arange(batch_size), idx, :32]
        backward_out = out3[:, 0, 32:]
        
        out_combined = torch.cat([forward_out, backward_out], dim=1)
        
        _, unsort_idx = sort_idx.sort()
        out_combined = out_combined[unsort_idx]
        
        out = self.fc1(out_combined)
        out = self.relu(out)
        out = self.dropout_fc(out)
        out = self.fc2(out)
        
        return out


def prepare_data_for_ebilstm(data_loader, feature_config, max_seq_len=200):
    all_features = []
    all_labels = []
    all_lengths = []
    
    for batch in data_loader:
        batch_size = batch.y.size(0)
        batch_assignment = batch.batch
        
        for i in range(batch_size):
            node_mask = (batch_assignment == i)
            node_features = batch.x[node_mask]
            
            if 'indices' in feature_config and len(feature_config['indices']) > 0:
                node_features = node_features[:, feature_config['indices']]
            
            seq_len = node_features.shape[0]
            feature_dim = node_features.shape[1]
            actual_len = min(seq_len, max_seq_len)
            
            if seq_len > max_seq_len:
                node_features = node_features[:max_seq_len]
            elif seq_len < max_seq_len:
                padding = torch.zeros(max_seq_len - seq_len, feature_dim)
                node_features = torch.cat([node_features, padding], dim=0)
            
            all_features.append(node_features)
            all_labels.append(batch.y[i].item())
            all_lengths.append(actual_len)
    
    features = torch.stack(all_features)
    labels = torch.tensor(all_labels, dtype=torch.long)
    lengths = torch.tensor(all_lengths, dtype=torch.long)
    
    return features, labels, lengths


def train_ebilstm_model(model, train_loader, val_loader, device, epochs=500, lr=0.001, patience=20):
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_val_auc = 0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_features, batch_labels, batch_lengths in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            batch_lengths = batch_lengths.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features, batch_lengths)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()
        
        # Validation
        model.eval()
        val_preds = []
        val_probs = []
        val_labels = []
        
        with torch.no_grad():
            for batch_features, batch_labels, batch_lengths in val_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                batch_lengths = batch_lengths.to(device)
                
                outputs = model(batch_features, batch_lengths)
                probs = F.softmax(outputs, dim=1)
                
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_probs.extend(probs[:, 1].cpu().numpy())
                val_labels.extend(batch_labels.cpu().numpy())
        
        val_auc = roc_auc_score(val_labels, val_probs) if len(set(val_labels)) > 1 else 0
        val_acc = accuracy_score(val_labels, val_preds)
        
        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            train_acc = train_correct / train_total
            print(f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Train Acc: {train_acc:.4f}, Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_val_auc


def evaluate_ebilstm_model(model, test_loader, device):
    model.eval()
    model = model.to(device)
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels, batch_lengths in test_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            batch_lengths = batch_lengths.to(device)
            
            outputs = model(batch_features, batch_lengths)
            probs = F.softmax(outputs, dim=1)
            
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'auc': auc,
        'accuracy': acc,
        'f1': f1,
        'mcc': mcc,
        'sensitivity': sensitivity,
        'specificity': specificity
    }


def train_single_fold(dataloaders, split_idx, feature_config, ebilstm_params, save_dir, device):
    config_name = feature_config['name']
    feature_dim = feature_config['feature_dim']
    
    np.random.seed(split_idx)
    torch.manual_seed(split_idx)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(split_idx)
    
    print(f"Stage 1: Pretraining")
    
    train_features, train_labels, train_lengths = prepare_data_for_ebilstm(
        dataloaders['pretrain_train'], feature_config, ebilstm_params['max_seq_len']
    )
    val_features, val_labels, val_lengths = prepare_data_for_ebilstm(
        dataloaders['pretrain_val'], feature_config, ebilstm_params['max_seq_len']
    )
    
    print(f"Pretrain: {train_features.shape[0]} samples, seq_len={train_features.shape[1]}, "
          f"features={train_features.shape[2]}, avg_len={train_lengths.float().mean():.1f}")
    
    train_dataset = TensorDataset(train_features, train_labels, train_lengths)
    val_dataset = TensorDataset(val_features, val_labels, val_lengths)
    train_loader = DataLoader(train_dataset, batch_size=ebilstm_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=ebilstm_params['batch_size'], shuffle=False)
    
    pretrain_model = EBiLSTMModel(
        input_dim=feature_dim,
        max_seq_len=ebilstm_params['max_seq_len']
    )
    
    pretrain_model, pretrain_val_auc = train_ebilstm_model(
        pretrain_model, train_loader, val_loader, device,
        epochs=ebilstm_params['epochs'],
        lr=ebilstm_params['lr'],
        patience=ebilstm_params['patience']
    )
    
    print(f"Pretrain VAL AUC: {pretrain_val_auc:.4f}")
    
    pretrain_model_path = os.path.join(save_dir, f'pretrain_{config_name}_split_{split_idx}.pth')
    torch.save(pretrain_model.state_dict(), pretrain_model_path)
    
    print(f"Stage 2: Fine-tuning")
    
    ft_train_features, ft_train_labels, ft_train_lengths = prepare_data_for_ebilstm(
        dataloaders['finetune_train'], feature_config, ebilstm_params['max_seq_len']
    )
    ft_val_features, ft_val_labels, ft_val_lengths = prepare_data_for_ebilstm(
        dataloaders['finetune_val'], feature_config, ebilstm_params['max_seq_len']
    )
    
    ft_train_dataset = TensorDataset(ft_train_features, ft_train_labels, ft_train_lengths)
    ft_val_dataset = TensorDataset(ft_val_features, ft_val_labels, ft_val_lengths)
    ft_train_loader = DataLoader(ft_train_dataset, batch_size=ebilstm_params['batch_size'], shuffle=True)
    ft_val_loader = DataLoader(ft_val_dataset, batch_size=ebilstm_params['batch_size'], shuffle=False)
    
    finetune_model = EBiLSTMModel(
        input_dim=feature_dim,
        max_seq_len=ebilstm_params['max_seq_len']
    )
    
    finetune_model, finetune_val_auc = train_ebilstm_model(
        finetune_model, ft_train_loader, ft_val_loader, device,
        epochs=ebilstm_params['epochs'],
        lr=ebilstm_params['lr'],
        patience=ebilstm_params['patience']
    )
    
    print(f"Finetune VAL AUC: {finetune_val_auc:.4f}")
    
    if finetune_val_auc > pretrain_val_auc:
        print(f"Using finetune model (AUC: {finetune_val_auc:.4f} > {pretrain_val_auc:.4f})")
        final_model_path = os.path.join(save_dir, f'final_{config_name}_split_{split_idx}.pth')
        torch.save(finetune_model.state_dict(), final_model_path)
        return final_model_path
    else:
        print(f"Using pretrain model (AUC: {pretrain_val_auc:.4f} >= {finetune_val_auc:.4f})")
        return pretrain_model_path


def test_single_fold(test_dataloader, model_path, feature_config, ebilstm_params, device):
    
    test_features, test_labels, test_lengths = prepare_data_for_ebilstm(
        test_dataloader, feature_config, ebilstm_params['max_seq_len']
    )
    
    test_dataset = TensorDataset(test_features, test_labels, test_lengths)
    test_loader = DataLoader(test_dataset, batch_size=ebilstm_params['batch_size'], shuffle=False)
    
    model = EBiLSTMModel(
        input_dim=feature_config['feature_dim'],
        max_seq_len=ebilstm_params['max_seq_len']
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    results = evaluate_ebilstm_model(model, test_loader, device)
    return results


def fair_evaluation_ebilstm_12_configs(
    data_dir='processed_data_XUAMP',
    n_splits=10,
    save_dir='evaluation_results_ebilstm_12_configs',
    skip_configs=None
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    all_feature_configs = [
        {'id': 1, 'name': 'OneHot_only', 'indices': np.arange(0, 20), 'feature_dim': 20,
         'description': 'One-hot encoding only'},
        {'id': 2, 'name': 'Position_only', 'indices': np.arange(20, 40), 'feature_dim': 20,
         'description': 'Position encoding only'},
        {'id': 3, 'name': 'ESM2_only', 'indices': np.arange(40, 80), 'feature_dim': 40,
         'description': 'ESM-2 embeddings only (40 dims)'},
        {'id': 4, 'name': 'ProtBERT_only', 'indices': np.arange(130, 258), 'feature_dim': 128,
         'description': 'ProtBERT embeddings only (128 dims)'},
        {'id': 5, 'name': 'AAC_only', 'indices': np.arange(80, 100), 'feature_dim': 20,
         'description': 'Amino Acid Composition only'},
        {'id': 6, 'name': 'PCP_only', 'indices': np.arange(100, 130), 'feature_dim': 30,
         'description': 'Physico-Chemical Properties only'},
        {'id': 7, 'name': 'OneHot_Position', 'indices': np.arange(0, 40), 'feature_dim': 40,
         'description': 'One-hot + Position encoding'},
        {'id': 8, 'name': 'AAC_PCP', 'indices': np.arange(80, 130), 'feature_dim': 50,
         'description': 'AAC + PCP'},
        {'id': 9, 'name': 'ESM2_ProtBERT',
         'indices': np.concatenate([np.arange(40, 80), np.arange(130, 258)]), 'feature_dim': 168,
         'description': 'ESM-2 + ProtBERT embeddings'},
        {'id': 10, 'name': 'Traditional_Compositional',
         'indices': np.concatenate([np.arange(0, 40), np.arange(80, 130)]), 'feature_dim': 90,
         'description': 'One-hot + Position + AAC + PCP (no embeddings)'},
        {'id': 11, 'name': 'Traditional_Embeddings',
         'indices': np.concatenate([np.arange(0, 80), np.arange(130, 258)]), 'feature_dim': 208,
         'description': 'One-hot + Position + ESM-2 + ProtBERT'},
        {'id': 12, 'name': 'Compositional_Embeddings',
         'indices': np.concatenate([np.arange(40, 130), np.arange(130, 258)]), 'feature_dim': 218,
         'description': 'AAC + PCP + ESM-2 + ProtBERT'}
    ]
    
    if skip_configs is None:
        skip_configs = []
    feature_configs = [cfg for cfg in all_feature_configs if cfg['id'] not in skip_configs]
    
    print(f"Testing {len(feature_configs)} feature configurations with EBiLSTM (skipping {len(skip_configs)} configs)")
    if skip_configs:
        print(f"Skipped configurations: {skip_configs}")
    print(f"Data directory: {data_dir}")
    
    print("\n" + "="*80)
    print("Loading all data with 258-dimensional features (with ProtBERT)...")
    all_positive_data, all_negative_data = load_all_xuamp_data(data_dir)
    
    print(f"Total positive samples: {len(all_positive_data)}")
    print(f"Total negative samples: {len(all_negative_data)}")
    
    if len(all_positive_data) > 0:
        sample_features = all_positive_data[0].x
        print(f"Original feature dimensions: {sample_features.shape}")
        if sample_features.shape[1] != 258:
            print(f"Warning: Expected 258 features, got {sample_features.shape[1]}")
    
    ebilstm_params = {
        'max_seq_len': 200,
        'batch_size': 32,
        'epochs': 500,
        'lr': 0.001,
        'patience': 20
    }
    
    all_feature_results = {}
    
    for feature_config in feature_configs:
        config_id = feature_config['id']
        config_name = feature_config['name']
        feature_dim = feature_config['feature_dim']
        
        print(f"\n{'='*80}")
        print(f"CONFIGURATION {config_id}: {config_name}")
        print(f"{'='*80}")
        print(f"Description: {feature_config['description']}")
        print(f"Expected dimensions: {feature_dim}")
        
        config_results = []
        
        for split_idx in range(n_splits):
            print(f"\nSplit {split_idx + 1}/{n_splits}")
            
            dataloaders = create_two_stage_random_split(
                all_positive_data, all_negative_data,
                test_ratio=0.2, random_seed=split_idx
            )
            
            try:
                best_model_path = train_single_fold(
                    dataloaders, split_idx, feature_config, ebilstm_params, save_dir, device
                )
                
                print(f"Testing on test set...")
                test_results = test_single_fold(
                    dataloaders['test'], best_model_path, feature_config, ebilstm_params, device
                )
                
                test_results['split'] = split_idx
                test_results['config_id'] = config_id
                test_results['config'] = config_name
                test_results['feature_dim'] = feature_dim
                config_results.append(test_results)
                
                print(f"AUC: {test_results['auc']:.4f}, "
                      f"Acc: {test_results['accuracy']:.4f}, "
                      f"F1: {test_results['f1']:.4f}")
                
                if os.path.exists(best_model_path):
                    os.remove(best_model_path)
                
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if config_results:
            all_feature_results[config_name] = pd.DataFrame(config_results)
            
            df = all_feature_results[config_name]
            print(f"\n{config_name} Summary:")
            print(f"AUC: {df['auc'].mean():.4f} +/- {df['auc'].std():.4f}")
            print(f"Accuracy: {df['accuracy'].mean():.4f} +/- {df['accuracy'].std():.4f}")
            print(f"F1: {df['f1'].mean():.4f} +/- {df['f1'].std():.4f}")
            print(f"MCC: {df['mcc'].mean():.4f} +/- {df['mcc'].std():.4f}")
    
    if all_feature_results:
        compare_all_configurations(all_feature_results, save_dir)
        save_detailed_results(all_feature_results, save_dir)
        return all_feature_results
    else:
        print("\nNo successful configurations")
        return None


def compare_all_configurations(all_results, save_dir):
    print(f"\n{'='*80}")
    print("FINAL COMPARISON OF ALL CONFIGURATIONS")
    print(f"{'='*80}")
    
    comparison_rows = []
    metrics = ['auc', 'accuracy', 'f1', 'mcc', 'sensitivity', 'specificity']
    
    for config_name, df in all_results.items():
        row = {
            'Configuration': config_name,
            'Config_ID': df['config_id'].iloc[0] if 'config_id' in df.columns else 'N/A',
            'Dimensions': df['feature_dim'].iloc[0] if 'feature_dim' in df.columns else 'N/A'
        }
        for metric in metrics:
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            row[f'{metric}_mean'] = mean_val
            row[f'{metric}_std'] = std_val
            row[metric.upper()] = f"{mean_val:.4f} +/- {std_val:.4f}"
        comparison_rows.append(row)
    
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df = comparison_df.sort_values('auc_mean', ascending=False)
    
    display_cols = ['Config_ID', 'Configuration', 'Dimensions'] + [m.upper() for m in metrics]
    print(comparison_df[display_cols].to_string(index=False))
    
    comparison_df.to_csv(os.path.join(save_dir, 'all_12_configs_comparison_ebilstm.csv'), index=False)
    
    
    best_idx = comparison_df['auc_mean'].idxmax()
    best_config = comparison_df.loc[best_idx, 'Configuration']
    best_auc = comparison_df.loc[best_idx, 'auc_mean']
    
    print(f"\nBEST CONFIGURATION: {best_config}")
    print(f"Config ID: {comparison_df.loc[best_idx, 'Config_ID']}")
    print(f"AUC: {best_auc:.4f}")
    print(f"Dimensions: {comparison_df.loc[best_idx, 'Dimensions']}")
    
    return comparison_df


def save_detailed_results(all_results, save_dir):
    all_data = []
    for config_name, df in all_results.items():
        df_copy = df.copy()
        df_copy['configuration'] = config_name
        all_data.append(df_copy)
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.to_csv(os.path.join(save_dir, 'all_12_configs_detailed_ebilstm.csv'), index=False)
    summary = combined_df.groupby(['config_id', 'configuration']).agg({
        'auc': ['mean', 'std', 'min', 'max'],
        'accuracy': ['mean', 'std'],
        'f1': ['mean', 'std'],
        'mcc': ['mean', 'std'],
        'sensitivity': ['mean', 'std'],
        'specificity': ['mean', 'std'],
        'feature_dim': 'first'
    }).round(4)
    summary.to_csv(os.path.join(save_dir, 'all_12_configs_summary_ebilstm.csv'))
    print(f"\nResults saved to {save_dir}")
    print("- all_12_configs_comparison_ebilstm.csv")
    print("- all_12_configs_detailed_ebilstm.csv")
    print("- all_12_configs_summary_ebilstm.csv")


if __name__ == '__main__':
    print("="*80)
    print("EBiLSTM - 12 FEATURE CONFIGURATIONS TEST")
    print("="*80)
    print("\nFeature space (258 dimensions total):")
    print("1. One-hot: 0-20")
    print("2. Position: 20-40")
    print("3. ESM-2: 40-80")
    print("4. AAC: 80-100")
    print("5. PCP: 100-130")
    print("6. ProtBERT: 130-258")
    print("\n12 Feature Configurations to test:")
    print("1.  One-hot only")
    print("2.  Position only")
    print("3.  ESM-2 only")
    print("4.  ProtBERT only")
    print("5.  AAC only")
    print("6.  PCP only")
    print("7.  One-hot + Position")
    print("8.  AAC + PCP")
    print("9.  ESM-2 + ProtBERT")
    print("10. One-hot + Position + AAC + PCP")
    print("11. One-hot + Position + ESM-2 + ProtBERT")
    print("12. AAC + PCP + ESM-2 + ProtBERT")
    print("="*80)
    
    skip_configs = []
    
    if skip_configs:
        print(f"\nSkipping configurations: {skip_configs}")
    print(f"Running {12 - len(skip_configs)} configurations")
    
    results = fair_evaluation_ebilstm_12_configs(
        data_dir='processed_data_XUAMP',
        n_splits=10,
        save_dir='evaluation_results_ebilstm_12_configs',
        skip_configs=skip_configs
    )
    
    if results:
        print(f"\n{'='*80}")
        print("EBiLSTM 12 FEATURE CONFIGURATIONS TEST COMPLETED")
        print(f"{'='*80}")
        print("\nConfiguration IDs tested:")
        for i in range(1, 13):
            status = "SKIPPED" if i in skip_configs else "TESTED"
            print(f"{i:2d}. {status}")
    else:
        print("\nTest failed")
