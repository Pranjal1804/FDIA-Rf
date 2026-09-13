import pandas as pd
import sys

try:
    df = pd.read_parquet('/home/pranjal/Desktop/fdia/data/simulated/grid2op_data.parquet')
    with open('/home/pranjal/Desktop/fdia/out.txt', 'w') as f:
        f.write(f"Shape: {df.shape}\n{df['label'].value_counts()}\n")
except Exception as e:
    with open('/home/pranjal/Desktop/fdia/out.txt', 'w') as f:
        f.write(f"Error: {str(e)}\n")
