"""
Train the model
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

# add src folder to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import *
from src.model import TransformerModel
from src.features import create_features, returns_to_percentiles

# use GPU if available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("Using CPU")

# create output folder
os.makedirs(RESULTS_DIR, exist_ok=True)


class MyDataset(Dataset):
    """Simple dataset class"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sequences(features_df, targets, lookback):
    """Create sequences for training"""
    X_list = []
    y_list = []
    
    for i in range(lookback, len(features_df) - 1):
        # check if target exists
        if features_df.index[i+1] not in targets.index:
            continue
        
        # get sequence
        seq = features_df.iloc[i-lookback:i].values
        
        # skip if has nan or inf
        if np.any(np.isnan(seq)) or np.any(np.isinf(seq)):
            continue
        
        # get target
        target = targets.loc[features_df.index[i+1]]
        if np.isnan(target):
            continue
        
        X_list.append(seq)
        y_list.append(target)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    return X, y


def train_model(X_train, y_train, X_val, y_val, input_dim):
    """Train the transformer model"""
    
    # set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # create model
    model = TransformerModel(input_dim)
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters")
    
    # optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.SmoothL1Loss()
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 15, 2)
    
    # data loaders
    train_dataset = MyDataset(X_train, y_train)
    val_dataset = MyDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # training loop
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(EPOCHS):
        # === TRAINING ===
        model.train()
        train_losses = []
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # forward pass
            optimizer.zero_grad()
            predictions = model(X_batch).squeeze()
            loss = criterion(predictions, y_batch)
            
            # add regularization for output distribution
            if len(predictions) > 1:
                pred_std = torch.std(predictions)
                target_std = torch.std(y_batch)
                loss = loss + 0.3 * torch.abs(pred_std - target_std)
            
            # backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        scheduler.step()
        
        # === VALIDATION ===
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                predictions = model(X_batch).squeeze()
                loss = criterion(predictions, y_batch)
                val_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        # print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} - train: {avg_train_loss:.5f}, val: {avg_val_loss:.5f}")
        
        # early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"Best validation loss: {best_val_loss:.5f}")
    
    return model


def main():
    print("=" * 60)
    print("TRAINING BTC TRANSFORMER")
    print("=" * 60)
    
    # === LOAD DATA ===
    print("\nLoading data...")
    
    btc = pd.read_csv(f"{CACHE_DIR}/btc_1h.csv", index_col=0, parse_dates=True)
    btc = btc[START_DATE:].sort_index()
    print(f"BTC: {len(btc):,} rows")
    
    # load optional data
    try:
        eth = pd.read_csv(f"{CACHE_DIR}/eth_1h.csv", index_col=0, parse_dates=True)
        eth = eth[START_DATE:].sort_index()
    except:
        eth = None
        print("ETH data not found")
    
    try:
        gold = pd.read_csv(f"{CACHE_DIR}/gold_1h.csv", index_col=0, parse_dates=True)
        gold = gold[START_DATE:].sort_index()
    except:
        gold = None
        print("Gold data not found")
    
    try:
        hashrate = pd.read_csv(f"{CACHE_DIR}/hash_rate.csv", index_col=0, parse_dates=True)
        hashrate = hashrate[START_DATE:].sort_index()
    except:
        hashrate = None
        print("Hash rate data not found")
    
    try:
        funding = pd.read_csv(f"{CACHE_DIR}/funding_1h.csv", index_col=0, parse_dates=True)
        funding = funding[START_DATE:].sort_index()
    except:
        funding = None
        print("Funding data not found")
    
    try:
        fear_greed = pd.read_csv(f"{CACHE_DAILY}/fear_greed.csv", index_col=0, parse_dates=True)
        fear_greed = fear_greed[START_DATE:].sort_index()
    except:
        fear_greed = None
        print("Fear & Greed data not found")
    
    # === SPLIT DATA ===
    train_btc = btc[:TRAIN_END]
    val_btc = btc[TRAIN_END:VAL_END]
    test_btc = btc[VAL_END:]
    
    print(f"\nTrain: {len(train_btc):,}, Val: {len(val_btc):,}, Test: {len(test_btc):,}")
    
    # === CREATE FEATURES ===
    print("\nCreating features...")
    
    # get data for each split
    train_eth = eth[:TRAIN_END] if eth is not None else None
    val_eth = eth[TRAIN_END:VAL_END] if eth is not None else None
    
    train_gold = gold[:TRAIN_END] if gold is not None else None
    val_gold = gold[TRAIN_END:VAL_END] if gold is not None else None
    
    train_funding = funding[:TRAIN_END] if funding is not None else None
    val_funding = funding[TRAIN_END:VAL_END] if funding is not None else None
    
    # create features
    feat_train = create_features(train_btc, train_eth, train_gold, hashrate, train_funding, fear_greed)
    feat_val = create_features(val_btc, val_eth, val_gold, hashrate, val_funding, fear_greed)
    
    # get common features
    common_features = list(set(feat_train.columns) & set(feat_val.columns))
    common_features.sort()
    
    print(f"Number of features: {len(common_features)}")
    
    # === NORMALIZE ===
    scaler = RobustScaler()
    
    train_scaled = scaler.fit_transform(feat_train[common_features])
    train_scaled = pd.DataFrame(train_scaled, index=feat_train.index, columns=common_features)
    
    val_scaled = scaler.transform(feat_val[common_features])
    val_scaled = pd.DataFrame(val_scaled, index=feat_val.index, columns=common_features)
    
    # === CREATE TARGETS ===
    returns = btc['close'].pct_change()
    
    train_percentiles = returns_to_percentiles(returns.loc[train_btc.index])
    val_percentiles = returns_to_percentiles(returns.loc[val_btc.index])
    
    # === CREATE SEQUENCES ===
    print("\nCreating sequences...")
    
    X_train, y_train = create_sequences(train_scaled, train_percentiles, LOOKBACK)
    X_val, y_val = create_sequences(val_scaled, val_percentiles, LOOKBACK)
    
    print(f"Train sequences: {X_train.shape[0]:,}")
    print(f"Val sequences: {X_val.shape[0]:,}")
    
    # === TRAIN ===
    print("\nTraining model...")
    
    model = train_model(X_train, y_train, X_val, y_val, X_train.shape[2])
    
    # === SAVE ===
    print("\nSaving model...")
    
    # save model
    model_path = f"{RESULTS_DIR}/best_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")
    
    # save scaler
    scaler_params = {
        'center_': scaler.center_.tolist(),
        'scale_': scaler.scale_.tolist(),
        'n_features_in_': int(scaler.n_features_in_),
        'feature_names': common_features
    }
    
    scaler_path = f"{RESULTS_DIR}/scaler.json"
    with open(scaler_path, 'w') as f:
        json.dump(scaler_params, f, indent=2)
    print(f"Saved scaler to {scaler_path}")
    
    # save config
    config_save = {
        'lookback': LOOKBACK,
        'd_model': D_MODEL,
        'n_heads': N_HEADS,
        'n_layers': N_LAYERS,
        'd_ff': D_FF,
        'dropout': DROPOUT,
        'n_features': len(common_features)
    }
    
    config_path = f"{RESULTS_DIR}/config.json"
    with open(config_path, 'w') as f:
        json.dump(config_save, f, indent=2)
    print(f"Saved config to {config_path}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()