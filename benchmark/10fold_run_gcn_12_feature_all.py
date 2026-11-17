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

from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from data_processing import load_all_xuamp_data, create_two_stage_random_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef, confusion_matrix

class SimpleGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.2):
        super(SimpleGCN, self).__init__()
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.dropouts.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))
        
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.dropouts.append(nn.Dropout(dropout))
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )
        
    def forward(self, x, edge_index, batch):
        for i, (conv, bn, dropout) in enumerate(zip(self.convs, self.bns, self.dropouts)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = dropout(x)
        
        x = global_mean_pool(x, batch)
        
        out = self.classifier(x)
        return out

def extract_feature_subset(data_list, feature_config):
    modified_data = []
    
    for data in data_list:
        if 'indices' in feature_config and len(feature_config['indices']) > 0:
            new_x = data.x[:, feature_config['indices']]
        else:
            new_x = data.x
        
        new_data = Data(
            x=new_x,
            edge_index=data.edge_index,
            y=data.y
        )
        modified_data.append(new_data)
    
    return modified_data

def check_data_statistics(data_list, config_name):
    if len(data_list) == 0:
        return
    
    sample_size = min(5, len(data_list))
    
    total_nodes = 0
    total_edges = 0
    edge_densities = []
    
    for i in range(sample_size):
        data = data_list[i]
        num_nodes = data.x.shape[0]
        num_edges = data.edge_index.shape[1]
        
        total_nodes += num_nodes
        total_edges += num_edges
        
        max_edges = num_nodes * (num_nodes - 1)
        if max_edges > 0:
            density = num_edges / max_edges
            edge_densities.append(density)
    
    print(f"Data statistics for {config_name} (sampled {sample_size} graphs):")
    print(f"Avg nodes per graph: {total_nodes/sample_size:.1f}")
    print(f"Avg edges per graph: {total_edges/sample_size:.1f}")
    if edge_densities:
        print(f"Avg edge density: {np.mean(edge_densities):.4f}")
    
    sample_features = data_list[0].x.numpy()
    print(f"Feature shape: {sample_features.shape}")
    print(f"Feature stats - Mean: {sample_features.mean():.4f}, Std: {sample_features.std():.4f}")

def train_gcn_model(model, train_loader, val_loader, device, epochs=100, lr=0.001, patience=20):
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
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = out.argmax(dim=1)
            train_correct += (pred == batch.y).sum().item()
            train_total += batch.y.size(0)
        
        # Validation
        model.eval()
        val_preds = []
        val_probs = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                probs = F.softmax(out, dim=1)
                
                val_preds.extend(out.argmax(dim=1).cpu().numpy())
                val_probs.extend(probs[:, 1].cpu().numpy())
                val_labels.extend(batch.y.cpu().numpy())
        
        # Calculate metrics
        val_auc = roc_auc_score(val_labels, val_probs) if len(set(val_labels)) > 1 else 0
        val_acc = accuracy_score(val_labels, val_preds)
        
        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
        
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_val_auc

def evaluate_gcn_model(model, test_loader, device):
    model.eval()
    model = model.to(device)
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            probs = F.softmax(out, dim=1)
            
            all_preds.extend(out.argmax(dim=1).cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    
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

def train_single_fold(dataloaders, split_idx, config_name, feature_dim, gcn_params, save_dir, device):
    
    np.random.seed(split_idx)
    torch.manual_seed(split_idx)
    
    print(f"Stage 1: Pretraining")
    
    # Initialize model
    pretrain_model = SimpleGCN(
        input_dim=feature_dim,
        **gcn_params
    )
    
    # Train
    pretrain_model, pretrain_val_auc = train_gcn_model(
        pretrain_model,
        dataloaders['pretrain_train'],
        dataloaders['pretrain_val'],
        device,
        epochs=100,
        lr=0.001,
        patience=20
    )
    
    print(f"Pretrain VAL AUC: {pretrain_val_auc:.4f}")
    
    pretrain_model_path = os.path.join(save_dir, f'pretrain_{config_name}_split_{split_idx}.pth')
    torch.save(pretrain_model.state_dict(), pretrain_model_path)
    
    print(f"Stage 2: Fine-tuning")
    
    finetune_model = SimpleGCN(
        input_dim=feature_dim,
        **gcn_params
    )
    
    finetune_model, finetune_val_auc = train_gcn_model(
        finetune_model,
        dataloaders['finetune_train'],
        dataloaders['finetune_val'],
        device,
        epochs=100,
        lr=0.001,
        patience=20
    )
    
    print(f"Finetune VAL AUC: {finetune_val_auc:.4f}")
    
    if finetune_val_auc > pretrain_val_auc:
        final_model_path = os.path.join(save_dir, f'final_{config_name}_split_{split_idx}.pth')
        torch.save(finetune_model.state_dict(), final_model_path)
        return final_model_path, feature_dim, gcn_params
    else:
        return pretrain_model_path, feature_dim, gcn_params

def test_single_fold(test_loader, model_path, feature_dim, gcn_params, device):
    
    model = SimpleGCN(input_dim=feature_dim, **gcn_params)
    model.load_state_dict(torch.load(model_path))
    
    results = evaluate_gcn_model(model, test_loader, device)
    return results

def fair_evaluation_gcn_12_configs(
    data_dir='processed_data_XUAMP',
    n_splits=10,
    save_dir='evaluation_results_gcn_12_configs',
    skip_configs=None
):

    # Set device
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
            'description': 'ESM2 + AAC + PCP + ProtBERT'
        }
    ]
    
    # Default configs to skip
    if skip_configs is None:
        skip_configs = []
    
    feature_configs = [cfg for cfg in all_feature_configs if cfg['id'] not in skip_configs]
    
    print(f"Testing {len(feature_configs)} feature configurations with GCN (skipping {len(skip_configs)} configs)")
    print(f"Skipped configurations: {skip_configs}")
    print(f"Data directory: {data_dir}")
    print("\n Note: Contact maps are loaded from {data_dir}/*/contact_maps/ directories")
    print("The AMPDataset class automatically loads contact maps and converts them to edges")
    
    print("\n" + "="*80)
    print("Loading all data with contact map edges...")
    all_positive_data, all_negative_data = load_all_xuamp_data(data_dir)
    
    print(f"Total positive samples: {len(all_positive_data)}")
    print(f"Total negative samples: {len(all_negative_data)}")
    
    if len(all_positive_data) > 0:
        sample_data = all_positive_data[0]
        print(f"\nSample graph statistics:")
        print(f"Number of nodes: {sample_data.x.shape[0]}")
        print(f"Number of edges: {sample_data.edge_index.shape[1]}")
        print(f"Feature dimensions: {sample_data.x.shape[1]}")
        
        num_nodes = sample_data.x.shape[0]
        num_edges = sample_data.edge_index.shape[1]
        max_possible_edges = num_nodes * (num_nodes - 1)
        if max_possible_edges > 0:
            edge_density = num_edges / max_possible_edges
            print(f"Edge density: {edge_density:.4f}")
            
            if edge_density == 1.0 or num_edges == num_nodes:
                print("Warning: Edge density suggests identity matrix (no real contact map)")
                print("This might mean contact maps weren't found and defaults were used")
            else:
                print("Contact maps appear to be loaded correctly")
    
    if len(all_positive_data) > 0:
        sample_features = all_positive_data[0].x
        print(f"\nOriginal feature dimensions: {sample_features.shape}")
        
        if sample_features.shape[1] != 258:
            print(f"Warning: Expected 258 features, got {sample_features.shape[1]}")
            
            if sample_features.shape[1] == 130:
                print("Detected 130 features - no ProtBERT")
                feature_configs = [cfg for cfg in feature_configs if cfg['id'] not in [4, 9, 11, 12]]
        
        # Display feature statistics
        print("\nOriginal feature statistics by component:")
        print(f"One-hot (0-20): mean={sample_features[:, 0:20].mean():.4f}, std={sample_features[:, 0:20].std():.4f}")
        print(f"Position (20-40): mean={sample_features[:, 20:40].mean():.4f}, std={sample_features[:, 20:40].std():.4f}")
        print(f"ESM-2 (40-80): mean={sample_features[:, 40:80].mean():.4f}, std={sample_features[:, 40:80].std():.4f}")
        print(f"AAC (80-100): mean={sample_features[:, 80:100].mean():.4f}, std={sample_features[:, 80:100].std():.4f}")
        print(f"PCP (100-130): mean={sample_features[:, 100:130].mean():.4f}, std={sample_features[:, 100:130].std():.4f}")
        if sample_features.shape[1] >= 258:
            print(f"ProtBERT (130-258): mean={sample_features[:, 130:258].mean():.4f}, std={sample_features[:, 130:258].std():.4f}")
    
    all_feature_results = {}
    
    gcn_params = {
        'hidden_dim': 128,
        'num_layers': 3,
        'dropout': 0.2
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
        print("Using contact map edges from data_processing module")
        
        try:
            modified_positive = extract_feature_subset(all_positive_data, feature_config)
            modified_negative = extract_feature_subset(all_negative_data, feature_config)
            
            if len(modified_positive) > 0:
                actual_dim = modified_positive[0].x.shape[1]
                print(f"Actual feature dimensions: {actual_dim}")
                
                check_data_statistics(modified_positive[:10], config_name)
                
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
            
            # Train and test
            try:
                model_path, feat_dim, params = train_single_fold(
                    dataloaders,
                    split_idx,
                    config_name,
                    actual_dim,
                    gcn_params,
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
            row[metric.upper()] = f"{mean_val:.4f} ± {std_val:.4f}"
        
        comparison_rows.append(row)
    
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df = comparison_df.sort_values('auc_mean', ascending=False)
    
    display_cols = ['Config_ID', 'Configuration', 'Dimensions'] + [m.upper() for m in metrics]
    print(comparison_df[display_cols].to_string(index=False))
    
    comparison_df.to_csv(os.path.join(save_dir, 'all_12_configs_comparison_gcn.csv'), index=False)
    
    best_idx = comparison_df['auc_mean'].idxmax()
    best_config = comparison_df.loc[best_idx, 'Configuration']
    best_auc = comparison_df.loc[best_idx, 'auc_mean']
    
    print(f"\n BEST GCN CONFIGURATION: {best_config}")
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
    
    combined_df.to_csv(os.path.join(save_dir, 'all_12_configs_detailed_gcn.csv'), index=False)
    
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
    
    summary.to_csv(os.path.join(save_dir, 'all_12_configs_summary_gcn.csv'))
    
    print(f"\n Results saved to {save_dir}")
    print("- all_12_configs_comparison_gcn.csv")
    print("- all_12_configs_detailed_gcn.csv")
    print("- all_12_configs_summary_gcn.csv")

if __name__ == '__main__':
    print("="*80)
    print("GCN - 12 FEATURE CONFIGURATIONS TEST (WITH CONTACT MAP EDGES)")
    print("="*80)
    print("\nFeature space (258 dimensions total):")
    print("1. One-hot: 0-20")
    print("2. Position: 20-40")
    print("3. ESM-2: 40-80")
    print("4. AAC: 80-100")
    print("5. PCP: 100-130")
    print("6. ProtBERT: 130-258")
    print("\n Contact maps are loaded from data directories and used as GRAPH EDGES")
    print("="*80)
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
    print("10. One-hot + Position + ESM-2 + AAC + PCP")
    print("11. One-hot + Position + ESM-2 + ProtBERT")
    print("12. ESM-2 + AAC + PCP + ProtBERT")
    print("="*80)
    
    # Configs to skip
    skip_configs = []
    
    if skip_configs:
        print(f"\nSkipping configurations: {skip_configs}")
    print(f"Running {12 - len(skip_configs)} configurations")
    
    results = fair_evaluation_gcn_12_configs(
        data_dir='processed_data_XUAMP',
        n_splits=10,
        save_dir='evaluation_results_gcn_12_configs',
        skip_configs=skip_configs
    )
    
    if results:
        print(f"\n{'='*80}")
        print("12 FEATURE CONFIGURATIONS TEST WITH GCN COMPLETED")
        print(f"{'='*80}")
        print("\nConfiguration IDs tested:")
        for i in range(1, 13):
            status = "SKIPPED" if i in skip_configs else "TESTED"
            print(f"{i:2d}. {status}")
    else:
        print("\n Test failed")