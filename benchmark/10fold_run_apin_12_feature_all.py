import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data
from data_processing import load_all_xuamp_data, create_two_stage_random_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef, confusion_matrix

# APIN Model
class APINModel(nn.Module):
    def __init__(self, input_dim, use_embedding=True, vocab_size=21, embed_dim=128, 
                 num_filters=64, filter_range=14, max_seq_len=200, dropout=0.2):
        
        super(APINModel, self).__init__()
        
        self.use_embedding = use_embedding
        self.num_filters = num_filters
        self.filter_range = filter_range
        self.max_seq_len = max_seq_len
        
        if use_embedding:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            conv_input_dim = embed_dim
        else:
            conv_input_dim = input_dim
            self.input_dim = input_dim
        
        # Multi-scale convolutional layers
        self.convs = nn.ModuleList()
        for i in range(filter_range):
            filter_size = (i + 1) * 2
            conv = nn.Conv1d(
                in_channels=conv_input_dim,
                out_channels=num_filters,
                kernel_size=filter_size,
                padding='same'
            )
            self.convs.append(conv)
        
        # Batch normalization
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_filters) for _ in range(filter_range)])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_filters, num_filters // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_filters // 2, 2)
        )
    
    def forward(self, x):
        if self.use_embedding:
            x = self.embedding(x)
        else:
            if len(x.shape) == 2:
                batch_size = x.shape[0]
                seq_len = min(self.max_seq_len, x.shape[1] // self.input_dim)
                if seq_len * self.input_dim < x.shape[1]:
                    x = x[:, :seq_len * self.input_dim]
                x = x.view(batch_size, seq_len, -1)
        
        x = x.transpose(1, 2)
        
        conv_outputs = []
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            conv_out = conv(x)
            conv_out = bn(conv_out)
            conv_out = F.relu(conv_out)
            
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))
            pooled = pooled.squeeze(2)
            conv_outputs.append(pooled)
        
        multi_scale_features = torch.stack(conv_outputs, dim=1)
        
        global_features, _ = torch.max(multi_scale_features, dim=1)
        
        # Dropout
        global_features = self.dropout(global_features)
        
        # Classification
        output = self.classifier(global_features)
        
        return output


def prepare_data_for_apin(data_loader, max_seq_len=200):
    all_features = []
    all_labels = []
    
    for batch in data_loader:
        batch_size = batch.y.size(0)
        
        batch_assignment = batch.batch
        
        for i in range(batch_size):
            node_mask = (batch_assignment == i)
            node_features = batch.x[node_mask]
        
            seq_len = node_features.shape[0]
            feature_dim = node_features.shape[1]
            
            if seq_len > max_seq_len:
                node_features = node_features[:max_seq_len]
            elif seq_len < max_seq_len:
                padding = torch.zeros(max_seq_len - seq_len, feature_dim)
                node_features = torch.cat([node_features, padding], dim=0)
            
            flat_features = node_features.flatten()
            
            all_features.append(flat_features)
            all_labels.append(batch.y[i].item())
    
    features = torch.stack(all_features)
    labels = torch.tensor(all_labels, dtype=torch.long)
    
    return features, labels


def train_apin_model(model, train_loader, val_loader, device, epochs=50, lr=0.0002, patience=20):
    model = model.to(device)
    
    optimizer = optim.RMSprop(model.parameters(), lr=lr)
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
        
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            
            loss.backward()
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
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_features)
                probs = F.softmax(outputs, dim=1)
                
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_probs.extend(probs[:, 1].cpu().numpy())
                val_labels.extend(batch_labels.cpu().numpy())
        
        # Calculate metrics
        val_auc = roc_auc_score(val_labels, val_probs) if len(set(val_labels)) > 1 else 0
        val_acc = accuracy_score(val_labels, val_preds)
        
        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            train_acc = train_correct / train_total
            print(f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Train Acc: {train_acc:.4f}, Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_val_auc


def evaluate_apin_model(model, test_loader, device):
    model.eval()
    model = model.to(device)
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_features)
            probs = F.softmax(outputs, dim=1)
            
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    # Calculate metrics
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


def train_single_fold(dataloaders, split_idx, config_name, feature_dim, apin_params, save_dir, device):
    
    np.random.seed(split_idx)
    torch.manual_seed(split_idx)
    
    print(f"Stage 1: Pretraining")
    
    train_features, train_labels = prepare_data_for_apin(
        dataloaders['pretrain_train'], 
        max_seq_len=apin_params['max_seq_len']
    )
    val_features, val_labels = prepare_data_for_apin(
        dataloaders['pretrain_val'],
        max_seq_len=apin_params['max_seq_len']
    )
    
    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    pretrain_model = APINModel(
        input_dim=feature_dim,
        use_embedding=False,
        num_filters=apin_params['num_filters'],
        filter_range=apin_params['filter_range'],
        max_seq_len=apin_params['max_seq_len'],
        dropout=apin_params['dropout']
    )
    
    # Train
    pretrain_model, pretrain_val_auc = train_apin_model(
        pretrain_model,
        train_loader,
        val_loader,
        device,
        epochs=apin_params['epochs'],
        lr=apin_params['lr'],
        patience=apin_params['patience']
    )
    
    print(f"Pretrain VAL AUC: {pretrain_val_auc:.4f}")
    
    # Save model
    pretrain_model_path = os.path.join(save_dir, f'pretrain_{config_name}_split_{split_idx}.pth')
    torch.save(pretrain_model.state_dict(), pretrain_model_path)
    
    print(f"Stage 2: Fine-tuning")
    
    finetune_train_features, finetune_train_labels = prepare_data_for_apin(
        dataloaders['finetune_train'],
        max_seq_len=apin_params['max_seq_len']
    )
    finetune_val_features, finetune_val_labels = prepare_data_for_apin(
        dataloaders['finetune_val'],
        max_seq_len=apin_params['max_seq_len']
    )
    
    finetune_train_dataset = TensorDataset(finetune_train_features, finetune_train_labels)
    finetune_val_dataset = TensorDataset(finetune_val_features, finetune_val_labels)
    
    finetune_train_loader = DataLoader(finetune_train_dataset, batch_size=32, shuffle=True)
    finetune_val_loader = DataLoader(finetune_val_dataset, batch_size=32, shuffle=False)
    
    finetune_model = APINModel(
        input_dim=feature_dim,
        use_embedding=False,
        num_filters=apin_params['num_filters'],
        filter_range=apin_params['filter_range'],
        max_seq_len=apin_params['max_seq_len'],
        dropout=apin_params['dropout']
    )
    
    finetune_model, finetune_val_auc = train_apin_model(
        finetune_model,
        finetune_train_loader,
        finetune_val_loader,
        device,
        epochs=apin_params['epochs'],
        lr=apin_params['lr'],
        patience=apin_params['patience']
    )
    
    print(f"Finetune VAL AUC: {finetune_val_auc:.4f}")
    
    if finetune_val_auc > pretrain_val_auc:
        final_model_path = os.path.join(save_dir, f'final_{config_name}_split_{split_idx}.pth')
        torch.save(finetune_model.state_dict(), final_model_path)
        return final_model_path, feature_dim, apin_params
    else:
        return pretrain_model_path, feature_dim, apin_params


def test_single_fold(test_loader, model_path, feature_dim, apin_params, device):
    test_features, test_labels = prepare_data_for_apin(
        test_loader,
        max_seq_len=apin_params['max_seq_len']
    )
    
    test_dataset = TensorDataset(test_features, test_labels)
    test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Load model
    model = APINModel(
        input_dim=feature_dim,
        use_embedding=False,
        num_filters=apin_params['num_filters'],
        filter_range=apin_params['filter_range'],
        max_seq_len=apin_params['max_seq_len'],
        dropout=apin_params['dropout']
    )
    model.load_state_dict(torch.load(model_path))
    
    results = evaluate_apin_model(model, test_data_loader, device)
    return results


def fair_evaluation_apin_12_configs(
    data_dir='processed_data_XUAMP',
    n_splits=10,
    save_dir='evaluation_results_apin_12_configs',
    skip_configs=None
):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Define all 12 feature configurations
    all_feature_configs = [
        {
            'id': 1,
            'name': 'OneHot_only',
            'indices': np.arange(0, 20),
            'feature_dim': 20,
            'description': 'One-hot encoding only'
        },
        {
            'id': 2,
            'name': 'Position_only',
            'indices': np.arange(20, 40),
            'feature_dim': 20,
            'description': 'Position encoding only'
        },
        {
            'id': 3,
            'name': 'ESM2_only',
            'indices': np.arange(40, 80),
            'feature_dim': 40,
            'description': 'ESM-2 embeddings only (40 dims)'
        },
        {
            'id': 4,
            'name': 'ProtBERT_only',
            'indices': np.arange(130, 258),
            'feature_dim': 128,
            'description': 'ProtBERT embeddings only (128 dims)'
        },
        {
            'id': 5,
            'name': 'AAC_only',
            'indices': np.arange(80, 100),
            'feature_dim': 20,
            'description': 'Amino Acid Composition only'
        },
        {
            'id': 6,
            'name': 'PCP_only',
            'indices': np.arange(100, 130),
            'feature_dim': 30,
            'description': 'Physico-Chemical Properties only'
        },
        {
            'id': 7,
            'name': 'OneHot_Position',
            'indices': np.arange(0, 40),
            'feature_dim': 40,
            'description': 'One-hot + Position encoding'
        },
        {
            'id': 8,
            'name': 'AAC_PCP',
            'indices': np.arange(80, 130),
            'feature_dim': 50,
            'description': 'AAC + PCP'
        },
        {
            'id': 9,
            'name': 'ESM2_ProtBERT',
            'indices': np.concatenate([np.arange(40, 80), np.arange(130, 258)]),
            'feature_dim': 168,
            'description': 'ESM-2 + ProtBERT embeddings'
        },
        {
            'id': 10,
            'name': 'Traditional_Compositional',
            'indices': np.concatenate([np.arange(0, 40), np.arange(80, 130)]),
            'feature_dim': 90,
            'description': 'One-hot + Position + AAC + PCP (no embeddings)'
        },
        {
            'id': 11,
            'name': 'Traditional_Embeddings',
            'indices': np.concatenate([np.arange(0, 80), np.arange(130, 258)]),
            'feature_dim': 208,
            'description': 'One-hot + Position + ESM-2 + ProtBERT'
        },
        {
            'id': 12,
            'name': 'Compositional_Embeddings',
            'indices': np.concatenate([np.arange(40, 130), np.arange(130, 258)]),
            'feature_dim': 218,
            'description': 'ESM-2 + AAC + PCP + ProtBERT'
        }
    ]
    
    # Default configs to skip
    if skip_configs is None:
        skip_configs = []
    
    feature_configs = [cfg for cfg in all_feature_configs if cfg['id'] not in skip_configs]
    
    print(f"Testing {len(feature_configs)} feature configurations with APIN (skipping {len(skip_configs)} configs)")
    print(f"Data directory: {data_dir}")
    print("Contact maps are NOT required for APIN")
    
    # Load all data with 258-dimensional features
    print("\n" + "="*80)
    print("Loading all data...")
    all_positive_data, all_negative_data = load_all_xuamp_data(data_dir)
    
    print(f"Total positive samples: {len(all_positive_data)}")
    print(f"Total negative samples: {len(all_negative_data)}")
    
    if len(all_positive_data) > 0:
        sample_features = all_positive_data[0].x
        print(f"\nOriginal feature dimensions: {sample_features.shape}")
        
        if sample_features.shape[1] != 258:
            print(f"Warning: Expected 258 features, got {sample_features.shape[1]}")
            
            if sample_features.shape[1] == 130:
                print("Detected 130 features - no ProtBERT")
                feature_configs = [cfg for cfg in feature_configs if cfg['id'] not in [4, 9, 11, 12]]
    
    # Store results
    all_feature_results = {}
    
    apin_params_base = {
        'num_filters': 64,
        'filter_range': 14,
        'max_seq_len': 200,
        'dropout': 0.2,
        'epochs': 50,
        'lr': 0.0002,
        'patience': 20
    }
    
    for feature_config in feature_configs:
        config_id = feature_config['id']
        config_name = feature_config['name']
        feature_dim = feature_config['feature_dim']
        
        print(f"\n{'='*80}")
        print(f"CONFIGURATION {config_id}: {config_name}")
        print(f"{'='*80}")
        print(f"Description: {feature_config['description']}")
        print(f"Expected dimensions: {feature_dim}")
        print(f"APIN Parameters: filters={apin_params_base['num_filters']}, "
              f"filter_range={apin_params_base['filter_range']}, "
              f"max_seq_len={apin_params_base['max_seq_len']}")
        
        apin_params = apin_params_base.copy()
        
        try:
            modified_positive = []
            for data in all_positive_data:
                new_data = Data(
                    x=data.x[:, feature_config['indices']],
                    edge_index=data.edge_index,
                    y=data.y
                )
                modified_positive.append(new_data)
            
            modified_negative = []
            for data in all_negative_data:
                new_data = Data(
                    x=data.x[:, feature_config['indices']],
                    edge_index=data.edge_index,
                    y=data.y
                )
                modified_negative.append(new_data)
            
            if len(modified_positive) > 0:
                actual_dim = modified_positive[0].x.shape[1]
                print(f"Actual feature dimensions: {actual_dim}")
                
        except Exception as e:
            print(f"Failed to extract features for {config_name}: {e}")
            continue
        
        config_results = []
        
        for split_idx in range(n_splits):
            print(f"\nSplit {split_idx + 1}/{n_splits}")
            
            dataloaders = create_two_stage_random_split(
                modified_positive,
                modified_negative,
                test_ratio=0.2,
                random_seed=split_idx
            )
            
            try:
                model_path, feat_dim, params = train_single_fold(
                    dataloaders,
                    split_idx,
                    config_name,
                    actual_dim,
                    apin_params,
                    save_dir,
                    device
                )
                
                test_results = test_single_fold(
                    dataloaders['test'],
                    model_path,
                    feat_dim,
                    params,
                    device
                )
                
                test_results['split'] = split_idx
                test_results['config_id'] = config_id
                test_results['config'] = config_name
                test_results['feature_dim'] = actual_dim
                config_results.append(test_results)
                
                print(f"AUC: {test_results['auc']:.4f}, "
                      f"Acc: {test_results['accuracy']:.4f}, "
                      f"F1: {test_results['f1']:.4f}")
                
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if config_results:
            all_feature_results[config_name] = pd.DataFrame(config_results)
            
            # Print summary
            df = all_feature_results[config_name]
            print(f"\n{config_name} Summary:")
            print(f"AUC: {df['auc'].mean():.4f} ± {df['auc'].std():.4f}")
            print(f"Accuracy: {df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f}")
            print(f"F1: {df['f1'].mean():.4f} ± {df['f1'].std():.4f}")
            print(f"MCC: {df['mcc'].mean():.4f} ± {df['mcc'].std():.4f}")
    
    # Compare all configurations
    if all_feature_results:
        compare_all_configurations(all_feature_results, save_dir)
        save_detailed_results(all_feature_results, save_dir)
        
        return all_feature_results
    else:
        print("\n No successful configurations")
        return None


def compare_all_configurations(all_results, save_dir):
    
    print(f"\n{'='*80}")
    print("FINAL COMPARISON OF ALL APIN CONFIGURATIONS")
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
            row[metric.upper()] = f"{mean_val:.4f} ± {std_val:.4f}"
        
        comparison_rows.append(row)
    
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df = comparison_df.sort_values('auc_mean', ascending=False)
    
    display_cols = ['Config_ID', 'Configuration', 'Dimensions'] + [m.upper() for m in metrics]
    print(comparison_df[display_cols].to_string(index=False))
    
    comparison_df.to_csv(os.path.join(save_dir, 'all_12_configs_comparison_apin.csv'), index=False)
    
    best_idx = comparison_df['auc_mean'].idxmax()
    best_config = comparison_df.loc[best_idx, 'Configuration']
    best_auc = comparison_df.loc[best_idx, 'auc_mean']
    
    print(f"\n BEST APIN CONFIGURATION: {best_config}")
    print(f"Config ID: {comparison_df.loc[best_idx, 'Config_ID']}")
    print(f"AUC: {best_auc:.4f}")
    print(f"Dimensions: {comparison_df.loc[best_idx, 'Dimensions']}")
    
    return comparison_df


def save_detailed_results(all_results, save_dir):
    
    # Combine all results
    all_data = []
    for config_name, df in all_results.items():
        df_copy = df.copy()
        df_copy['configuration'] = config_name
        all_data.append(df_copy)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    combined_df.to_csv(os.path.join(save_dir, 'all_12_configs_detailed_apin.csv'), index=False)
    
    # Create summary statistics
    summary = combined_df.groupby(['config_id', 'configuration']).agg({
        'auc': ['mean', 'std', 'min', 'max'],
        'accuracy': ['mean', 'std'],
        'f1': ['mean', 'std'],
        'mcc': ['mean', 'std'],
        'sensitivity': ['mean', 'std'],
        'specificity': ['mean', 'std'],
        'feature_dim': 'first'
    }).round(4)
    
    summary.to_csv(os.path.join(save_dir, 'all_12_configs_summary_apin.csv'))
    
    print(f"\n Results saved to {save_dir}")
    print("- all_12_configs_comparison_apin.csv")
    print("- all_12_configs_detailed_apin.csv")
    print("- all_12_configs_summary_apin.csv")


if __name__ == '__main__':
    print("="*80)
    print("APIN - 12 FEATURE CONFIGURATIONS TEST")
    print("="*80)
    print("\nFeature space (258 dimensions total):")
    print("1. One-hot: 0-20")
    print("2. Position: 20-40")
    print("3. ESM-2: 40-80")
    print("4. AAC: 80-100")
    print("5. PCP: 100-130")
    print("6. ProtBERT: 130-258")
    print("\n12 Feature Configurations to test:")
    print("1. One-hot only")
    print("2. Position only")
    print("3. ESM-2 only")
    print("4. ProtBERT only")
    print("5. AAC only")
    print("6. PCP only")
    print("7. One-hot + Position")
    print("8. AAC + PCP")
    print("9. ESM-2 + ProtBERT")
    print("10. One-hot + Position + AAC + PCP")
    print("11. One-hot + Position + ESM-2 + ProtBERT")
    print("12. ESM-2 + AAC + PCP + ProtBERT")
    print("="*80)
    
    # Configs to skip
    skip_configs = []
    
    if skip_configs:
        print(f"\nSkipping configurations: {skip_configs}")
    print(f"Running {12 - len(skip_configs)} configurations")
    
    results = fair_evaluation_apin_12_configs(
        data_dir='processed_data_enhanced_XUAMP',
        n_splits=10,
        save_dir='evaluation_results_apin_12_configs',
        skip_configs=skip_configs
    )
    
    if results:
        print(f"\n{'='*80}")
        print("12 FEATURE CONFIGURATIONS TEST WITH APIN COMPLETED")
        print(f"{'='*80}")
        print("\nConfiguration IDs tested:")
        for i in range(1, 13):
            status = "SKIPPED" if i in skip_configs else "TESTED"
            print(f"{i:2d}. {status}")
    else:
        print("\n Test failed")
