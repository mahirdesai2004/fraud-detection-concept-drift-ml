import os
import pandas as pd

def download_and_save_data(output_path="data/raw_data.csv"):
    urls = [
        "https://huggingface.co/datasets/electricsheepafrica/nigerian-banking-mobile-money/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
    ]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    for url in urls:
        print(f"Trying to download from {url}...")
        try:
            df = pd.read_parquet(url)
            df.to_csv(output_path, index=False)
            print(f"Success! Saved to {output_path} with {len(df)} rows.")
            return
        except Exception as e:
            print(f"Failed: {e}")

if __name__ == "__main__":
    download_and_save_data()
