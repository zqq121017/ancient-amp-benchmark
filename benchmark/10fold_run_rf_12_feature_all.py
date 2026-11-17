import os
import numpy as np
import torch
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from torch_geometric.data import Data, DataLoader
from data_processing import load_all_xuamp_data, create_two_stage_random_split
from models_rf.RandomForestAdvanced import RFModelAdvanced
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef, confusion_matrix

def load_contact_maps(data_dir, dataset_type):
    contact_map_dir = os.path.join(data_dir, dataset_type, 'contact_maps')
    contact_maps = {}
    
    if os.path.exists(contact_map_dir):
        for file in os.listdir(contact_map_dir):
            if file.endswith('.npz'):
                file_path = os.path.join(contact_map_dir, file)
                data = np.load(file_path)
                seq_id = file.replace('.npz', '')
                contact_maps[seq_id] = data['contact_map']
    
    return contact_maps

def flatten_contact_map(contact_map, max_length=100):
    upper_tri = np.triu(contact_map, k=1)
    
    flattened = upper_tri.flatten()
    
    expected_size = (max_length * (max_length - 1)) // 2
    
    if len(flattened) > expected_size:
        return flattened[:expected_size]
    else:
        padded = np.zeros(expected_size)
        padded[:len(flattened)] = flattened
        return padded

def extract_feature_subset(data_list, feature_config, contact_maps=None):
    modified_data = []
    
    for i, data in enumerate(data_list):
        if feature_config.get('use_contact_map', False) and contact_maps is not None:
            seq_len = data.x.shape[0]
            
            if i < len(contact_maps):
                contact_map = list(contact_maps.values())[i]
                
                contact_features = flatten_contact_map(contact_map)
                
                contact_features_repeated = np.tile(contact_features.reshape(1, -1), (seq_len, 1))
                
                if 'indices' in feature_config and len(feature_config['indices']) > 0:
                    node_features = data.x[:, feature_config['indices']]
                    combined_features = np.concatenate([node_features.numpy(), contact_features_repeated], axis=1)
                else:
                    combined_features = contact_features_repeated
                
                new_x = torch.tensor(combined_features, dtype=torch.float32)
            else:
                if 'indices' in feature_config and len(feature_config['indices']) > 0:
                    new_x = data.x[:, feature_config['indices']]
                else:
                    new_x = torch.zeros((seq_len, 1))
        else:
            if 'indices' in feature_config and len(feature_config['indices']) > 0:
                new_x = data.x[:, feature_config['indices']]
            else:
                new_x = data.x
        
        new_data = Data(
            x=new_x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr if hasattr(data, 'edge_attr') else None,
            y=data.y
        )
        modified_data.append(new_data)
    
    return modified_data

def fair_evaluation_rf_12_configs(
    data_dir='processed_data_XUAMP',
    n_splits=10,
    save_dir='evaluation_results_rf_12_configs',
    skip_configs=None
):

    os.makedirs(save_dir, exist_ok=True)
    
    # Define all 12 feature configurations
    all_feature_configs = [
        {
            'id': 1,
            'name': 'OneHot_only',
            'indices': np.arange(0, 20),
            'feature_dim': 20,
            'description': 'One-hot encoding only',
            'use_contact_map': False
        },
        {
            'id': 2,
            'name': 'Position_only',
            'indices': np.arange(20, 40),
            'feature_dim': 20,
            'description': 'Position encoding only',
            'use_contact_map': False
        },
        {
            'id': 3,
            'name': 'ESM2_only',
            'indices': np.arange(40, 80),
            'feature_dim': 40,
            'description': 'ESM-2 embeddings only (40 dims)',
            'use_contact_map': False
        },
        {
            'id': 4,
            'name': 'ProtBERT_only',
            'indices': np.arange(130, 258),
            'feature_dim': 128,
            'description': 'ProtBERT embeddings only (128 dims)',
            'use_contact_map': False
        },
        {
            'id': 5,
            'name': 'AAC_only',
            'indices': np.arange(80, 100),
            'feature_dim': 20,
            'description': 'Amino Acid Composition only',
            'use_contact_map': False
        },
        {
            'id': 6,
            'name': 'PCP_only',
            'indices': np.arange(100, 130),
            'feature_dim': 30,
            'description': 'Physico-Chemical Properties only',
            'use_contact_map': False
        },
        {
            'id': 7,
            'name': 'OneHot_Position',
            'indices': np.arange(0, 40),
            'feature_dim': 40,
            'description': 'One-hot + Position encoding',
            'use_contact_map': False
        },
        {
            'id': 8,
            'name': 'AAC_PCP',
            'indices': np.arange(80, 130),
            'feature_dim': 50,
            'description': 'AAC + PCP',
            'use_contact_map': False
        },
        {
            'id': 9,
            'name': 'ESM2_ProtBERT',
            'indices': np.concatenate([np.arange(40, 80), np.arange(130, 258)]),
            'feature_dim': 168,
            'description': 'ESM-2 + ProtBERT embeddings',
            'use_contact_map': False
        },
        {
            'id': 10,
            'name': 'Traditional_Compositional',
            'indices': np.concatenate([np.arange(0, 40), np.arange(80, 130)]),
            'feature_dim': 90,
            'description': 'One-hot + Position + AAC + PCP (no embeddings)',
            'use_contact_map': False
        },
        {
            'id': 11,
            'name': 'Traditional_Embeddings',
            'indices': np.concatenate([np.arange(0, 80), np.arange(130, 258)]),
            'feature_dim': 208,
            'description': 'One-hot + Position + ESM-2 + ProtBERT',
            'use_contact_map': False
        },
        {
            'id': 12,
            'name': 'Compositional_Embeddings',
            'indices': np.concatenate([np.arange(40, 130), np.arange(130, 258)]),
            'feature_dim': 218,
            'description': 'AAC + PCP + ESM-2 + ProtBERT',
            'use_contact_map': False
        }
    ]
    
    # Default configs to skip
    if skip_configs is None:
        skip_configs = []
    
    feature_configs = [cfg for cfg in all_feature_configs if cfg['id'] not in skip_configs]
    
    print(f"Testing {len(feature_configs)} feature configurations (skipping {len(skip_configs)} configs)")
    if skip_configs:
        print(f"Skipped configurations: {skip_configs}")
    print(f"Data directory: {data_dir}")
    
    print("\n" + "="*80)
    print("Loading all data with 258-dimensional features...")
    all_positive_data, all_negative_data = load_all_xuamp_data(data_dir)
    
    print(f"Total positive samples: {len(all_positive_data)}")
    print(f"Total negative samples: {len(all_negative_data)}")
    
    contact_maps = None
    if any(cfg.get('use_contact_map', False) for cfg in feature_configs):
        print("\nLoading contact maps...")
        contact_maps = {}
    
    if len(all_positive_data) > 0:
        sample_features = all_positive_data[0].x
        print(f"Original feature dimensions: {sample_features.shape}")
        
        if sample_features.shape[1] != 258:
            print(f"Warning: Expected 258 features, got {sample_features.shape[1]}")
            
            if sample_features.shape[1] == 130:
                print("Detected 130 features - adjusting for dataset WITHOUT ProtBERT")
                for config in feature_configs:
                    if config['id'] in [4, 9, 11, 12]:  # Configs that use ProtBERT
                        print(f"Skipping config {config['id']}: {config['name']} (requires ProtBERT)")
                        feature_configs.remove(config)
        
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
    
    for feature_config in feature_configs:
        config_id = feature_config['id']
        config_name = feature_config['name']
        feature_dim = feature_config['feature_dim']
        
        print(f"\n{'='*80}")
        print(f"CONFIGURATION {config_id}: {config_name}")
        print(f"{'='*80}")
        print(f"Description: {feature_config['description']}")
        print(f"Expected dimensions: {feature_dim}")
        if feature_config.get('use_contact_map', False):
            print("Note: Using contact maps as additional features")
        
        try:
            modified_positive = extract_feature_subset(all_positive_data, feature_config, contact_maps)
            modified_negative = extract_feature_subset(all_negative_data, feature_config, contact_maps)
            
            if len(modified_positive) > 0:
                actual_dim = modified_positive[0].x.shape[1]
                print(f"Actual feature dimensions: {actual_dim}")
                
                # Print feature statistics
                sample_features = modified_positive[0].x.numpy()
                print(f"Feature statistics:")
                print(f"Mean: {sample_features.mean():.4f}")
                print(f"Std: {sample_features.std():.4f}")
                print(f"Min: {sample_features.min():.4f}")
                print(f"Max: {sample_features.max():.4f}")
                print(f"% zeros: {(sample_features == 0).mean() * 100:.2f}%")
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
                rf_params = {
                    'model_type': 'rf',
                    'n_estimators': 300,
                    'max_depth': 20,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'max_features': 'sqrt' if actual_dim > 10 else min(actual_dim, 5),
                    'optimize_hyperparams': False
                }
                
                model_path = train_single_fold(
                    dataloaders,
                    split_idx,
                    config_name,
                    actual_dim,
                    rf_params,
                    save_dir
                )
                
                test_results = test_single_fold(
                    dataloaders['test'],
                    model_path
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

def train_single_fold(dataloaders, split_idx, config_name, feature_dim, rf_params, save_dir):
    
    np.random.seed(split_idx)
    torch.manual_seed(split_idx)
    
    print(f"Stage 1: Pretraining")
    
    pretrain_model = RFModelAdvanced(
        node_feature_dim=feature_dim,
        random_state=split_idx,
        **rf_params
    )
    
    # Train
    pretrain_model.fit(dataloaders['pretrain_train'], verbose=False)
    
    pretrain_model_path = os.path.join(save_dir, f'pretrain_{config_name}_split_{split_idx}.pkl')
    pretrain_model.save_model(pretrain_model_path)
    
    # Evaluate
    pretrain_val_auc = evaluate_model(pretrain_model, dataloaders['pretrain_val'])
    print(f"Pretrain VAL AUC: {pretrain_val_auc:.4f}")
    
    print(f"Stage 2: Fine-tuning")
    
    finetune_model = RFModelAdvanced(
        node_feature_dim=feature_dim,
        random_state=split_idx + 1000,
        **rf_params
    )
    
    finetune_model.fit(dataloaders['finetune_train'], verbose=False)
    
    # Evaluate
    finetune_val_auc = evaluate_model(finetune_model, dataloaders['finetune_val'])
    print(f"Finetune VAL AUC: {finetune_val_auc:.4f}")
    
    if finetune_val_auc > pretrain_val_auc:
        final_model_path = os.path.join(save_dir, f'final_{config_name}_split_{split_idx}.pkl')
        finetune_model.save_model(final_model_path)
    else:
        final_model_path = pretrain_model_path
    
    return final_model_path

def evaluate_model(model, dataloader):
    predictions, probabilities = model.predict(dataloader)
    
    y_true = []
    for data in dataloader:
        y_true.extend(data.y.cpu().numpy())
    
    auc = roc_auc_score(y_true, probabilities[:, 1])
    return auc

def test_single_fold(test_loader, model_path):
    
    model = RFModelAdvanced.load_model(model_path)
    predictions, probabilities = model.predict(test_loader)
    
    y_true = []
    for data in test_loader:
        y_true.extend(data.y.cpu().numpy())
    
    y_true = np.array(y_true)
    y_pred = predictions
    probs = probabilities[:, 1]
    
    # Calculate metrics
    auc = roc_auc_score(y_true, probs)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
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
    
    comparison_df.to_csv(os.path.join(save_dir, 'all_12_configs_comparison.csv'), index=False)
    
    best_idx = comparison_df['auc_mean'].idxmax()
    best_config = comparison_df.loc[best_idx, 'Configuration']
    best_auc = comparison_df.loc[best_idx, 'auc_mean']
    
    print(f"\n BEST CONFIGURATION: {best_config}")
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
    
    combined_df.to_csv(os.path.join(save_dir, 'all_12_configs_detailed.csv'), index=False)
    
    summary = combined_df.groupby(['config_id', 'configuration']).agg({
        'auc': ['mean', 'std', 'min', 'max'],
        'accuracy': ['mean', 'std'],
        'f1': ['mean', 'std'],
        'mcc': ['mean', 'std'],
        'sensitivity': ['mean', 'std'],
        'specificity': ['mean', 'std'],
        'feature_dim': 'first'
    }).round(4)
    
    summary.to_csv(os.path.join(save_dir, 'all_12_configs_summary.csv'))
    
    print(f"\n Results saved to {save_dir}")
    print("- all_12_configs_comparison.csv")
    print("- all_12_configs_detailed.csv")
    print("- all_12_configs_summary.csv")

if __name__ == '__main__':
    print("="*80)
    print("RANDOM FOREST - 12 FEATURE CONFIGURATIONS TEST")
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
    print("12. AAC + PCP + ESM-2 + ProtBERT")
    print("="*80)
    
    skip_configs = []
    
    if skip_configs:
        print(f"\nSkipping configurations: {skip_configs}")
    print(f"Running {12 - len(skip_configs)} configurations")
    
    results = fair_evaluation_rf_12_configs(
        data_dir='processed_data_XUAMP',
        n_splits=10,
        save_dir='evaluation_results_rf_12_configs',
        skip_configs=skip_configs
    )
    
    if results:
        print(f"\n{'='*80}")
        print("12 FEATURE CONFIGURATIONS TEST COMPLETED")
        print(f"{'='*80}")
        print("\nConfiguration IDs tested:")
        for i in range(1, 13):
            status = "SKIPPED" if i in skip_configs else "TESTED"
            print(f"{i:2d}. {status}")
    else:
        print("\n Test failed")
