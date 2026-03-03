import pandas as pd
import glob
from argparse import ArgumentParser

def aggregate_results(folder_path: str):
    files = glob.glob(f'{folder_path}/*.csv')
    aggregated_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    aggregated_df.to_csv(f'clean_results/aggregated.csv', index=False)

if __name__ == "__main__":
    parser = ArgumentParser(description="Aggregate CSV results from a folder.")
    parser.add_argument('--folder', type=str, required=True, help='Path to folder containing CSV files')
    args = parser.parse_args()

    aggregate_results(args.folder)