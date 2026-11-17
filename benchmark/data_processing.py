import os
import numpy as np
import torch
from torch_geometric.data import Data, Dataset, DataLoader
import random
from sklearn.model_selection import train_test_split

class AMPDataset(Dataset):
    def __init__(self, root, features_dir, contact_maps_dir, transform=None, pre_transform=None):
        super(AMPDataset, self).__init__(root, transform, pre_transform)
        self.features_dir = features_dir
        self.contact_maps_dir = contact_maps_dir
        self.file_ids = [f.split('.')[0] for f in os.listdir(features_dir) if f.endswith('.npz')]
        
    def len(self):
        return len(self.file_ids)
    
    def get(self, idx):
        file_id = self.file_ids[idx]
        
        features_path = os.path.join(self.features_dir, f"{file_id}.npz")
        features_data = np.load(features_path)
        
        contact_map_npz_path = os.path.join(self.contact_maps_dir, f"{file_id}.npz")
        contact_map_npy_path = os.path.join(self.contact_maps_dir, f"{file_id}.npy")
        
        if os.path.exists(contact_map_npz_path):
            contact_map_data = np.load(contact_map_npz_path)
            contact_map = contact_map_data['contact_map']
        elif os.path.exists(contact_map_npy_path):
            contact_map = np.load(contact_map_npy_path)
        else:
            print(f"Warning: Contact map not found for {file_id}, using default")
            seq_len = features_data['features'].shape[0]
            contact_map = np.eye(seq_len)
        
        node_features = features_data['features']  # Shape: [L, feature_dim]
        
        if 'label' in features_data:
            label = features_data['label']
        else:
            if 'pos' in self.features_dir:
                label = 1
            elif 'neg' in self.features_dir:
                label = 0
            else:
                label = 0
        
        # Convert to PyTorch tensors
        x = torch.tensor(node_features, dtype=torch.float)
        
        if isinstance(label, np.ndarray):
            if label.size == 1:
                y = torch.tensor(int(label.item()), dtype=torch.long)
            else:
                y = torch.tensor(int(label[0]), dtype=torch.long)
        else:
            y = torch.tensor(int(label), dtype=torch.long)
        
        # Create edge_index from contact map
        edge_index = self._contact_map_to_edge_index(contact_map)
        
        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, y=y)
        
        return data
    
    def _contact_map_to_edge_index(self, contact_map):
        indices = np.where(contact_map > 0)
        edge_index = torch.tensor(np.array([indices[0], indices[1]]), dtype=torch.long)
        return edge_index

def load_datasets(base_dir):
    datasets = {}
    dataloaders = {}
    
    dataset_keys = [
        'test_pos', 'test_neg',
        'pretrain_train_pos', 'pretrain_train_neg',
        'pretrain_val_pos', 'pretrain_val_neg',
        'train_pos', 'train_neg',
        'val_pos', 'val_neg'
    ]
    
    for key in dataset_keys:
        features_dir = os.path.join(base_dir, key, 'features')
        contact_maps_dir = os.path.join(base_dir, key, 'contact_maps')
        
        if os.path.exists(features_dir) and os.path.exists(contact_maps_dir):
            datasets[key] = AMPDataset(
                root=os.path.join(base_dir, key),
                features_dir=features_dir,
                contact_maps_dir=contact_maps_dir
            )
            
    pretrain_train_data = torch.utils.data.ConcatDataset([
        datasets['pretrain_train_pos'], 
        datasets['pretrain_train_neg']
    ])
    
    pretrain_val_data = torch.utils.data.ConcatDataset([
        datasets['pretrain_val_pos'], 
        datasets['pretrain_val_neg']
    ])
    
    train_data = torch.utils.data.ConcatDataset([
        datasets['train_pos'], 
        datasets['train_neg']
    ])
    
    val_data = torch.utils.data.ConcatDataset([
        datasets['val_pos'], 
        datasets['val_neg']
    ])
    
    test_data = torch.utils.data.ConcatDataset([
        datasets['test_pos'], 
        datasets['test_neg']
    ])
    
    batch_size = 512
    
    dataloaders = {
        'pretrain_train': DataLoader(pretrain_train_data, batch_size=batch_size, shuffle=True),
        'pretrain_val': DataLoader(pretrain_val_data, batch_size=batch_size),
        'train': DataLoader(train_data, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_data, batch_size=batch_size),
        'test': DataLoader(test_data, batch_size=batch_size)
    }
    
    return dataloaders

def load_all_xuamp_data(base_dir):
    all_positive_data = []
    all_negative_data = []
    
    positive_keys = ['test_pos', 'pretrain_train_pos', 'pretrain_val_pos', 'train_pos', 'val_pos']
    negative_keys = ['test_neg', 'pretrain_train_neg', 'pretrain_val_neg', 'train_neg', 'val_neg']
    
    for key in positive_keys:
        features_dir = os.path.join(base_dir, key, 'features')
        contact_maps_dir = os.path.join(base_dir, key, 'contact_maps')
        
        if os.path.exists(features_dir):
            if not os.path.exists(contact_maps_dir):
                print(f"Warning: Contact maps directory not found for {key}, will use default contact maps")
            
            dataset = AMPDataset(
                root=os.path.join(base_dir, key),
                features_dir=features_dir,
                contact_maps_dir=contact_maps_dir if os.path.exists(contact_maps_dir) else features_dir
            )
            for i in range(len(dataset)):
                try:
                    all_positive_data.append(dataset[i])
                except Exception as e:
                    print(f"Error loading positive sample {i} from {key}: {e}")
                    continue
    
    for key in negative_keys:
        features_dir = os.path.join(base_dir, key, 'features')
        contact_maps_dir = os.path.join(base_dir, key, 'contact_maps')
        
        if os.path.exists(features_dir):
            if not os.path.exists(contact_maps_dir):
                print(f"Warning: Contact maps directory not found for {key}, will use default contact maps")
            
            dataset = AMPDataset(
                root=os.path.join(base_dir, key),
                features_dir=features_dir,
                contact_maps_dir=contact_maps_dir if os.path.exists(contact_maps_dir) else features_dir
            )
            for i in range(len(dataset)):
                try:
                    all_negative_data.append(dataset[i])
                except Exception as e:
                    print(f"Error loading negative sample {i} from {key}: {e}")
                    continue
    
    return all_positive_data, all_negative_data

def create_random_split(all_positive_data, all_negative_data, test_ratio=0.2, random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
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
    
    train_data = pos_train + neg_train
    test_data = pos_test + neg_test
    
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    train_loader = DataLoader(train_data, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=512, shuffle=False)
    
    return train_loader, test_loader


def create_two_stage_random_split(all_positive_data, all_negative_data, test_ratio=0.2, random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
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
    
    pos_pretrain, pos_finetune = train_test_split(
        pos_train, 
        test_size=0.3, 
        random_state=random_seed + 1
    )
    
    neg_pretrain, neg_finetune = train_test_split(
        neg_train, 
        test_size=0.3, 
        random_state=random_seed + 1
    )
    
    pos_pretrain_train, pos_pretrain_val = train_test_split(
        pos_pretrain, 
        test_size=0.2, 
        random_state=random_seed + 2
    )
    
    neg_pretrain_train, neg_pretrain_val = train_test_split(
        neg_pretrain, 
        test_size=0.2, 
        random_state=random_seed + 2
    )
    
    pos_finetune_train, pos_finetune_val = train_test_split(
        pos_finetune, 
        test_size=0.2, 
        random_state=random_seed + 3
    )
    
    neg_finetune_train, neg_finetune_val = train_test_split(
        neg_finetune, 
        test_size=0.2, 
        random_state=random_seed + 3
    )
    
    pretrain_train_data = pos_pretrain_train + neg_pretrain_train
    pretrain_val_data = pos_pretrain_val + neg_pretrain_val
    finetune_train_data = pos_finetune_train + neg_finetune_train
    finetune_val_data = pos_finetune_val + neg_finetune_val
    test_data = pos_test + neg_test
    
    random.shuffle(pretrain_train_data)
    random.shuffle(pretrain_val_data)
    random.shuffle(finetune_train_data)
    random.shuffle(finetune_val_data)
    random.shuffle(test_data)
    
    dataloaders = {
        'pretrain_train': DataLoader(pretrain_train_data, batch_size=512, shuffle=True),
        'pretrain_val': DataLoader(pretrain_val_data, batch_size=512, shuffle=False),
        'finetune_train': DataLoader(finetune_train_data, batch_size=512, shuffle=True),
        'finetune_val': DataLoader(finetune_val_data, batch_size=512, shuffle=False),
        'test': DataLoader(test_data, batch_size=512, shuffle=False)
    }
    
    return dataloaders
