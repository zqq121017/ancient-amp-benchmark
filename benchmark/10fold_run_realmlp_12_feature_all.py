import os
import sys
import numpy as np
import torch
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pytabkit-main'))

from torch_geometric.data import Data, DataLoader
from data_processing import load_all_xuamp_data
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef, confusion_matrix
from sklearn.model_selection import train_test_split

try:
    import pytorch_lightning
    import torchmetrics
except ImportError as e:
    print("Missing required dependencies")
    sys.exit(1)

try:
    import pytabkit
    RealMLP_TD_Classifier = pytabkit.RealMLP_TD_Classifier
except (ImportError, AttributeError):
    try:
        from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_Classifier
    except ImportError as e:
        print(f"Failed to import RealMLP: {e}")
        sys.exit(1)

def create_simple_split(all_positive_data, all_negative_data, test_ratio=0.2, random_seed=42):
    pos_train, pos_test = train_test_split(
        all_positive_data, 
        test_size=test_ratio, 
        random_state=random_seed
    )
    
    neg_train, neg_test = train_test_split(
        all_negative_data, 
        test_size=test_ratio, 
        random_state=random_seed
    )
    
    return {
        'train_pos': pos_train,
        'train_neg': neg_train,
        'test_pos': pos_test,
        'test_neg': neg_test
    }

def extract_feature_subset(data_list, feature_config):
    features = []
    labels = []
    
    for data in data_list:
        if 'indices' in feature_config and len(feature_config['indices']) > 0:
            node_features = data.x[:, feature_config['indices']].numpy()
        else:
            node_features = data.x.numpy()
        
        seq_features = np.mean(node_features, axis=0)
        features.append(seq_features)
        labels.append(data.y.item() if hasattr(data.y, 'item') else data.y.numpy()[0])
    
    return np.array(features), np.array(labels)

def train_realmlp_model(X_train, y_train, X_test, y_test, feature_dim, config_name, device='cuda', verbosity=0):
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        if verbosity > 0:
            print(f"CUDA not available, using CPU")
    
    model = RealMLP_TD_Classifier(
        device=device,
        random_state=42,
        n_cv=1,
        n_epochs=256,
        batch_size=256,
        hidden_sizes=[256, 256, 256],
        val_fraction=0.2,
        verbosity=verbosity,
        val_metric_name='cross_entropy',
        use_ls=False,
        p_drop=0.2,
        lr=0.001,
        wd=1e-5,
        use_early_stopping=True,
        early_stopping_additive_patience=20,
        n_threads=4
    )
    
    try:
        model.fit(X_train, y_train)
        
        test_probs = model.predict_proba(X_test)[:, 1]
        test_preds = model.predict(X_test)
        
        test_auc = roc_auc_score(y_test, test_probs) if len(set(y_test)) > 1 else 0
        test_acc = accuracy_score(y_test, test_preds)
        test_f1 = f1_score(y_test, test_preds, average='binary')
        test_mcc = matthews_corrcoef(y_test, test_preds)
        
        tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics = {
            'auc': test_auc,
            'accuracy': test_acc,
            'f1': test_f1,
            'mcc': test_mcc,
            'sensitivity': sensitivity,
            'specificity': specificity
        }
        
        return model, metrics
        
    except Exception as e:
        print(f"Error training RealMLP: {e}")
        import traceback
        if verbosity > 0:
            traceback.print_exc()
        return None, None

def fair_evaluation_realmlp_12_configs(
    data_dir='processed_data_XUAMP',
    n_splits=10,
    save_dir='evaluation_results_realmlp_12_configs',
    skip_configs=None,
    device='cuda',
    verbosity=0
):

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
            'description': 'AAC + PCP + ESM-2 + ProtBERT'
        }
    ]
    
    # Default configs to skip
    if skip_configs is None:
        skip_configs = []
    
    feature_configs = [cfg for cfg in all_feature_configs if cfg['id'] not in skip_configs]
    
    print(f"Testing {len(feature_configs)} feature configurations with RealMLP (skipping {len(skip_configs)} configs)")
    if skip_configs:
        print(f"Skipped configurations: {skip_configs}")
    print(f"Data directory: {data_dir}")
    print(f"Device: {device}")
    
    print("\n" + "="*80)
    print("Loading all data...")
    all_positive_data, all_negative_data = load_all_xuamp_data(data_dir)
    
    print(f"Total positive samples: {len(all_positive_data)}")
    print(f"Total negative samples: {len(all_negative_data)}")
    
    if len(all_positive_data) > 0:
        sample_features = all_positive_data[0].x
        print(f"Original feature dimensions: {sample_features.shape}")
        feature_dim = sample_features.shape[1]
        
        if feature_dim == 258:
            print("Detected 258 features (with ProtBERT)")
        elif feature_dim == 130:
            print("Detected 130 features (without ProtBERT)")
            # Adjust configs that use ProtBERT
            feature_configs = [cfg for cfg in feature_configs if cfg['id'] not in [4, 9, 11, 12]]
            print(f"Adjusted to {len(feature_configs)} configurations")
        else:
            print(f"Unexpected feature dimension: {feature_dim}")
    
    all_feature_results = {}
    
    for feature_config in feature_configs:
        config_id = feature_config['id']
        config_name = feature_config['name']
        expected_dim = feature_config['feature_dim']
        
        print(f"\n{'='*80}")
        print(f"CONFIGURATION {config_id}: {config_name}")
        print(f"{'='*80}")
        print(f"Description: {feature_config['description']}")
        print(f"Expected dimensions: {expected_dim}")
        
        config_results = []
        
        for split_idx in range(n_splits):
            print(f"\nSplit {split_idx + 1}/{n_splits}")
            
            data_split = create_simple_split(
                all_positive_data,
                all_negative_data,
                test_ratio=0.2,
                random_seed=split_idx
            )
            
            train_positive = data_split['train_pos']
            train_negative = data_split['train_neg']
            test_positive = data_split['test_pos']
            test_negative = data_split['test_neg']
            
            train_data = train_positive + train_negative
            test_data = test_positive + test_negative
            
            try:
                X_train, y_train = extract_feature_subset(train_data, feature_config)
                X_test, y_test = extract_feature_subset(test_data, feature_config)
                
                actual_dim = X_train.shape[1]
                if split_idx == 0:
                    print(f"Actual feature dimensions: {actual_dim}")
                    if actual_dim != expected_dim:
                        print(f"Warning: Expected {expected_dim}, got {actual_dim}")
                
                model, test_metrics = train_realmlp_model(
                    X_train, y_train,
                    X_test, y_test,
                    actual_dim, config_name,
                    device=device,
                    verbosity=0
                )
                
                if model is None:
                    print(f"Failed to train model for split {split_idx + 1}")
                    continue
                
                # Store results
                split_results = {
                    'config_id': config_id,
                    'split': split_idx + 1,
                    'feature_dim': actual_dim,
                    'auc': test_metrics['auc'],
                    'accuracy': test_metrics['accuracy'],
                    'f1': test_metrics['f1'],
                    'mcc': test_metrics['mcc'],
                    'sensitivity': test_metrics['sensitivity'],
                    'specificity': test_metrics['specificity']
                }
                
                config_results.append(split_results)
                
                print(f"Test AUC: {test_metrics['auc']:.4f}, "
                      f"Accuracy: {test_metrics['accuracy']:.4f}, "
                      f"F1: {test_metrics['f1']:.4f}")
                
            except Exception as e:
                print(f"Error in split {split_idx + 1}: {e}")
                if verbosity > 0:
                    import traceback
                    traceback.print_exc()
                continue
        
        if config_results:
            config_df = pd.DataFrame(config_results)
            all_feature_results[config_name] = config_df
            
            config_df.to_csv(
                os.path.join(save_dir, f'config_{config_id:02d}_{config_name}_results.csv'),
                index=False
            )
            
            # Print summary
            print(f"\nSummary for {config_name}:")
            print(f"Mean AUC: {config_df['auc'].mean():.4f} ± {config_df['auc'].std():.4f}")
            print(f"Mean Accuracy: {config_df['accuracy'].mean():.4f} ± {config_df['accuracy'].std():.4f}")
            print(f"Mean F1: {config_df['f1'].mean():.4f} ± {config_df['f1'].std():.4f}")
            print(f"Mean MCC: {config_df['mcc'].mean():.4f} ± {config_df['mcc'].std():.4f}")
    
    if all_feature_results:
        create_comparison_summary(all_feature_results, save_dir)
        save_detailed_results(all_feature_results, save_dir)
    
    return all_feature_results

def create_comparison_summary(all_results, save_dir):
    
    summary_data = []
    
    for config_name, df in all_results.items():
        summary_data.append({
            'configuration': config_name,
            'config_id': df['config_id'].iloc[0],
            'feature_dim': df['feature_dim'].iloc[0],
            'auc_mean': df['auc'].mean(),
            'auc_std': df['auc'].std(),
            'accuracy_mean': df['accuracy'].mean(),
            'accuracy_std': df['accuracy'].std(),
            'f1_mean': df['f1'].mean(),
            'f1_std': df['f1'].std(),
            'mcc_mean': df['mcc'].mean(),
            'mcc_std': df['mcc'].std(),
            'sensitivity_mean': df['sensitivity'].mean(),
            'sensitivity_std': df['sensitivity'].std(),
            'specificity_mean': df['specificity'].mean(),
            'specificity_std': df['specificity'].std()
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('auc_mean', ascending=False)
    
    # Save summary
    summary_df.to_csv(os.path.join(save_dir, 'realmlp_12_configs_comparison.csv'), index=False)
    
    print(f"\n{'='*80}")
    print("TOP CONFIGURATIONS BY AUC (RealMLP):")
    print(f"{'='*80}")
    
    for idx, row in summary_df.head(5).iterrows():
        print(f"{row['configuration']:30s} | AUC: {row['auc_mean']:.4f} ± {row['auc_std']:.4f} | "
              f"Acc: {row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f} | "
              f"F1: {row['f1_mean']:.4f} ± {row['f1_std']:.4f}")

def save_detailed_results(all_results, save_dir):
    
    # Combine all results
    all_data = []
    for config_name, df in all_results.items():
        df_copy = df.copy()
        df_copy['configuration'] = config_name
        all_data.append(df_copy)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    combined_df.to_csv(os.path.join(save_dir, 'realmlp_12_configs_detailed.csv'), index=False)
    
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
    
    summary.to_csv(os.path.join(save_dir, 'realmlp_12_configs_summary.csv'))
    
    comparison_format = []
    for config_name, df in all_results.items():
        comparison_format.append({
            'configuration': config_name,
            'config_id': df['config_id'].iloc[0],
            'feature_dim': df['feature_dim'].iloc[0],
            'auc_mean': df['auc'].mean(),
            'auc_std': df['auc'].std(),
            'accuracy_mean': df['accuracy'].mean(),
            'accuracy_std': df['accuracy'].std(),
            'f1_mean': df['f1'].mean(),
            'f1_std': df['f1'].std(),
            'mcc_mean': df['mcc'].mean(),
            'mcc_std': df['mcc'].std()
        })
    comparison_df = pd.DataFrame(comparison_format)
    comparison_df.to_csv(os.path.join(save_dir, 'all_12_configs_comparison.csv'), index=False)
    
    print(f"\n Results saved to {save_dir}")
    print("- realmlp_12_configs_comparison.csv")
    print("- realmlp_12_configs_detailed.csv")
    print("- realmlp_12_configs_summary.csv")
    print("- all_12_configs_comparison.csv (for comparison with RF/GCN)")

if __name__ == '__main__':
    print("="*80)
    print("REALMLP - 12 FEATURE CONFIGURATIONS TEST")
    print("="*80)
    
    skip_configs = []
    
    # Configure device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    verbosity = 0
    
    if skip_configs:
        print(f"\nSkipping configurations: {skip_configs}")
    print(f"Running {12 - len(skip_configs)} configurations")
    print(f"Device: {device}")
    
    results = fair_evaluation_realmlp_12_configs(
        data_dir='processed_data_XUAMP',
        n_splits=10,
        save_dir='evaluation_results_realmlp_12_configs',
        skip_configs=skip_configs,
        device=device,
        verbosity=verbosity
    )
    
    if results:
        print(f"\n{'='*80}")
        print("REALMLP 12 FEATURE CONFIGURATIONS TEST COMPLETED")
        print(f"{'='*80}")
        print("\nConfiguration IDs tested:")
        for i in range(1, 13):
            status = "SKIPPED" if i in skip_configs else "TESTED"
            print(f"{i:2d}. {status}")
        
    else:
        print("\n Test failed")
