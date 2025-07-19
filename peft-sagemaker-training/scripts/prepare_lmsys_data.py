import pandas as pd
import json
from sklearn.model_selection import train_test_split

def prepare_lmsys_data():
    df = pd.read_csv('lmsys-annotations-v1.csv')
    
    winner_cols = ['winner_model_a', 'winner_model_b', 'winner_tie']
    for col in winner_cols:
        df[col] = df[col].fillna(0).astype(int)
    
    row_sums = df[winner_cols].sum(axis=1)
    df_clean = df[row_sums == 1].copy()
    
    # Create labels: 0=Model A wins, 1=Model B wins, 2=Tie
    def get_label(row):
        if row['winner_model_a'] == 1:
            return 0
        elif row['winner_model_b'] == 1:
            return 1
        else:  # winner_tie == 1
            return 2
    
    df_clean['label'] = df_clean.apply(get_label, axis=1)
    df_clean['text'] = df_clean['prompt'] + '<Model A>: ' + df_clean['response_a'] + '<Model B>: ' + df_clean['response_b']
    
    # Keep only what we need
    data = df_clean[['text', 'label']].to_dict('records')
    
    # Split data (80/20)
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Save to local files
    with open('data/local/train.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open('data/local/validation.json', 'w') as f:
        json.dump(val_data, f, indent=2)
    
    print(f"✅ Created {len(train_data)} train samples, {len(val_data)} validation samples")
    print(f"📊 Class distribution: {pd.Series([d['label'] for d in data]).value_counts().to_dict()}")

if __name__ == "__main__":
    prepare_lmsys_data()