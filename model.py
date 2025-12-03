import os
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import xgboost as xgb
import catboost as cb


#SEED_GLOBAL = 42

# random.seed(SEED_GLOBAL)

# np.random.seed(SEED_GLOBAL)


# torch.manual_seed(43)

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# try:

#     torch.use_deterministic_algorithms(True, warn_only=True)
# except TypeError:

#     torch.use_deterministic_algorithms(True)


BOARD_ROWS = 6
BOARD_COLS = 7

def get_board_from_flat(flat_array):

    board = np.array(flat_array).reshape(BOARD_ROWS, BOARD_COLS)

    if not board.flags.writeable:
        board = board.copy()
    return board

def get_next_open_row(board, col):

    for r in range(BOARD_ROWS):
        if board[r][col] == 0:
            return r
    return -1

def check_win(board, piece):

    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS - 3):
            if np.all(board[r, c:c+4] == piece): 
                return True

    for r in range(BOARD_ROWS - 3):
        for c in range(BOARD_COLS):
            if np.all(board[r:r+4, c] == piece): 
                return True

    for r in range(BOARD_ROWS - 3):
        for c in range(BOARD_COLS - 3):
            if board[r, c] == piece and board[r+1, c+1] == piece and \
               board[r+2, c+2] == piece and board[r+3, c+3] == piece: 
                return True

    for r in range(3, BOARD_ROWS):
        for c in range(BOARD_COLS - 3):
            if board[r, c] == piece and board[r-1, c+1] == piece and \
               board[r-2, c+2] == piece and board[r-3, c+3] == piece: 
                return True
    return False

def count_threats(board, piece, playable_only=False):

    threat_count = 0
    odd_playable_threats = 0
    even_playable_threats = 0
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            for dr, dc in directions:
                if not (0 <= r + 3*dr < BOARD_ROWS and 0 <= c + 3*dc < BOARD_COLS):
                    continue
                window = [board[r + i*dr][c + i*dc] for i in range(4)]
                if window.count(piece) == 3 and window.count(0) == 1:
                    if not playable_only:
                        threat_count += 1
                        continue
                    empty_slot_index = window.index(0)
                    empty_r = r + empty_slot_index * dr
                    empty_c = c + empty_slot_index * dc
                    playable_row_in_col = get_next_open_row(board, empty_c)
                    if playable_row_in_col == empty_r:
                        if empty_r % 2 == 1:
                            odd_playable_threats += 1
                        else:
                            even_playable_threats += 1
    if not playable_only:
        return threat_count
    else:
        return odd_playable_threats, even_playable_threats

def count_split_threes(board, piece):

    count = 0
    patterns = [[piece, 0, piece, piece], [piece, piece, 0, piece]]
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            for dr, dc in [(0, 1), (1, 1), (1, -1)]:
                if 0 <= r + 3*dr < BOARD_ROWS and 0 <= c + 3*dc < BOARD_COLS:
                    window = [board[r + i*dr][c + i*dc] for i in range(4)]
                    if window in patterns: 
                        count += 1
    return count

def create_features(df_row):

    flat_board = df_row[[f'p{i}' for i in range(1, 43)]].values.astype(int)
    board = flat_board.reshape(6, 7)
    turn = df_row['turn']
    opp = -1 * turn
    
    feats = {}
    feats['my_odd_threats'], feats['my_even_threats'] = count_threats(board, turn, playable_only=True)
    feats['opp_odd_threats'], feats['opp_even_threats'] = count_threats(board, opp, playable_only=True)
    feats['my_split_threes'] = count_split_threes(board, turn)
    feats['opp_split_threes'] = count_split_threes(board, opp)
    feats['center_col_pieces'] = np.count_nonzero(board[:, 3] == turn)
    
    feats['winning_moves'] = 0
    feats['blocking_moves'] = 0
    feats['suicide_moves'] = 0
    feats['attack_moves'] = 0
    feats['fork_moves'] = 0
    
    opponent_winning_cols = set()
    for c in range(7):
        r = get_next_open_row(board, c)
        if r != -1:
            board[r][c] = opp
            if check_win(board, opp): 
                opponent_winning_cols.add(c)
            board[r][c] = 0
    
    p_threats_before = feats['my_odd_threats'] + feats['my_even_threats']
    move_scores = []
    valid_moves = 0
    
    for c in range(7):
        r = get_next_open_row(board, c)
        if r == -1:
            move_scores.append(-1000)
            continue
        valid_moves += 1
        score = 0
        
        board[r][c] = turn
        if check_win(board, turn):
            feats['winning_moves'] += 1
            score += 1000
        
        p_odd, p_even = count_threats(board, turn, playable_only=True)
        if (p_odd + p_even) > p_threats_before:
            feats['attack_moves'] += 1
            score += 50
            if (p_odd + p_even) >= p_threats_before + 2:
                feats['fork_moves'] += 1
                score += 200
        
        board[r][c] = 0
        
        if r + 1 < BOARD_ROWS:
            board[r][c] = turn
            board[r+1][c] = opp
            if check_win(board, opp):
                feats['suicide_moves'] += 1
                score -= 500
            board[r+1][c] = 0
            board[r][c] = 0
        
        if c in opponent_winning_cols:
            feats['blocking_moves'] += 1
            score += 500
        
        score += [0, 1, 2, 4, 2, 1, 0][c]
        move_scores.append(score)
    
    feats['valid_moves_count'] = valid_moves
    feats['best_move_score'] = max(move_scores) if valid_moves > 0 else -1000
    feats['missed_blocks'] = len(opponent_winning_cols) - feats['blocking_moves']
    
    return pd.Series(feats)

def expand_board_channels(df):

    df_out = df.copy()
    p_cols = [f'p{i}' for i in range(1, 43)]
    
    player_ids = df['turn'].values.reshape(-1, 1)
    board_values = df[p_cols].values
    
    my_board = (board_values == player_ids).astype(int)
    opp_board = (board_values == (-1 * player_ids)).astype(int)
    
    df_my = pd.DataFrame(my_board, columns=[f'my_{c}' for c in p_cols], index=df.index)
    df_opp = pd.DataFrame(opp_board, columns=[f'opp_{c}' for c in p_cols], index=df.index)
    
    return pd.concat([df_out.drop(columns=p_cols), df_my, df_opp], axis=1)


class ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(128)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class OutBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(128, 3, kernel_size=1)
        self.bn = nn.BatchNorm2d(3)
        self.fc1 = nn.Linear(3 * 6 * 7, 128)
        self.fc2 = nn.Linear(128, 7)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(-1, 3 * 6 * 7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class DeepConnect4Net(nn.Module):
    def __init__(self, num_blocks=8):
        super().__init__()
        self.conv = ConvBlock()
        self.res_blocks = nn.ModuleList([ResBlock() for _ in range(num_blocks)])
        self.out = OutBlock()

    def forward(self, x):
        x = self.conv(x)
        for block in self.res_blocks:
            x = block(x)
        return self.out(x)

class Model:
    def __init__(self, artifacts_dir="submission_artifacts3"):

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.artifacts_dir = os.path.join(self.base_dir, artifacts_dir)
        
        print(f"Loading artifacts from: {self.artifacts_dir}")
        

        self.train_cols = joblib.load(os.path.join(self.artifacts_dir, 'training_columns.pkl'))
        print(f"Loaded {len(self.train_cols)} training columns")
        

        self.scaler_tree = joblib.load(os.path.join(self.artifacts_dir, 'scaler_tree.pkl'))
        self.scaler_nn = joblib.load(os.path.join(self.artifacts_dir, 'scaler_nn.pkl'))
        print(f"Tree scaler features: {self.scaler_tree.n_features_in_}")
        print(f"NN scaler features: {self.scaler_nn.n_features_in_}")
        

        self.cat = cb.CatBoostClassifier()
        self.cat.load_model(os.path.join(self.artifacts_dir, 'base_catboost.cbm'))
        print("✓ Loaded CatBoost")
        
        self.xgb = joblib.load(os.path.join(self.artifacts_dir, 'base_xgboost.pkl'))
        print("✓ Loaded XGBoost")
        
        self.rf = joblib.load(os.path.join(self.artifacts_dir, 'base_rf.pkl'))
        print("✓ Loaded Random Forest")
        

        self.device = torch.device("cpu")
        self.cnn = DeepConnect4Net(num_blocks=8).to(self.device)
        self.cnn.load_state_dict(torch.load(
            os.path.join(self.artifacts_dir, 'deep_resnet_seed_43.pth'),
            map_location=self.device
        ))
        self.cnn.eval()
        print("✓ Loaded CNN")
        

        self.meta = joblib.load(os.path.join(self.artifacts_dir, 'meta_stacker.pkl'))
        print("✓ Loaded Meta-Stacker")
        
        print("\nAll models loaded successfully!")
    
    def _prepare_input_data(self, df):


        df_feat = pd.concat([df, df.apply(create_features, axis=1)], axis=1)
        

        df_expanded = expand_board_channels(df_feat)
        

        df_aligned = df_expanded.reindex(columns=self.train_cols, fill_value=0)
        
        # Log alignment info
        missing = set(self.train_cols) - set(df_expanded.columns)
        extra = set(df_expanded.columns) - set(self.train_cols)
        
        if missing:
            print(f"{len(missing)} columns missing, filled with 0")
        if extra:
            print(f"{len(extra)} extra columns dropped")
        
        return df_aligned
    
    def _prepare_cnn_input(self, df_aligned):

        my_cols = [c for c in df_aligned.columns if 'my_p' in c]
        opp_cols = [c for c in df_aligned.columns if 'opp_p' in c]
        
        my_plane = df_aligned[my_cols].values.reshape(-1, 6, 7).astype(np.float32)
        opp_plane = df_aligned[opp_cols].values.reshape(-1, 6, 7).astype(np.float32)
        

        valid_mask = np.zeros_like(my_plane, dtype=np.float32)
        for i in range(len(my_plane)):
            top_row = my_plane[i, 0, :] + opp_plane[i, 0, :]
            valid_mask[i, :, :] = (top_row == 0).astype(np.float32)
        

        img_array = np.stack([my_plane, opp_plane, valid_mask], axis=1)
        img_tensor = torch.tensor(img_array, dtype=torch.float32)
        
        return img_tensor
    
    def predict(self, df):

        df_aligned = self._prepare_input_data(df)
        

        X_tree = self.scaler_tree.transform(df_aligned)
        

        p_cat = self.cat.predict_proba(X_tree)
        p_xgb = self.xgb.predict_proba(X_tree)
        p_rf = self.rf.predict_proba(X_tree)
        

        img_tensor = self._prepare_cnn_input(df_aligned)
        with torch.no_grad():
            logits = self.cnn(img_tensor)
            p_cnn = F.softmax(logits, dim=1).numpy()
        

        X_meta = np.hstack([p_cat, p_xgb, p_rf, p_cnn, X_tree])
        

        final_probs = self.meta.predict_proba(X_meta)
        

        p_cols_top = [f'p{i}' for i in range(36, 43)]
        if all(col in df.columns for col in p_cols_top):
            top_vals = df[p_cols_top].values
            mask = (top_vals == 0).astype(float)
            
            masked_probs = final_probs * mask
            row_sums = masked_probs.sum(axis=1)

            mask_failed = row_sums == 0
            if mask_failed.any():
                masked_probs[mask_failed] = final_probs[mask_failed]
            
            final_probs = masked_probs
        

        predictions = np.argmax(final_probs, axis=1)
        return predictions
    

if __name__ == "__main__":

    # random.seed(SEED_GLOBAL)
    # np.random.seed(SEED_GLOBAL)
    # torch.manual_seed(43)
    

    import sys
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    

    print("Initializing ensemble model...")
    try:
        model = Model()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nMake sure you have saved all artifacts using save_artifacts_from_notebook()")
        raise
    

    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_path = os.path.join(base_dir, "test.csv")

    if os.path.exists(test_path):
        test_data = pd.read_csv(test_path)
        print(f"\nLoaded test data: {len(test_data)} rows")
        
        X_test = test_data.drop(columns=['id'])
        predictions = model.predict(X_test)
        

        submission = pd.DataFrame({
            "id": range(1, len(predictions) + 1),
            "label_move_col": predictions
        })
        
        output_path = "submission_final.csv"
        submission.to_csv(output_path, index=False)
        print(f"\nPredictions saved to: {output_path}")
        print(f"First 10 predictions: {predictions[:10]}")
        
    else: print("failed to locate test.csv, make sure it's in the same folder as model.py")