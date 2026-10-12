import os
import pandas as pd
from nasbench import api

# 1. Initialize the NAS-Bench-101 API
# Change this path to where your .tfrecord file is stored
tfrecord_path = r'F:\vinh\vscs\fi\datas\nasbench_full.tfrecord'
output_csv_path = 'nasbench101_flat_dataset.csv'

print("Loading NAS-Bench-101 dataset (this may take a couple of minutes)...")
nasbench = api.NASBench(tfrecord_path)

print("Starting extraction to CSV...")
buffer = []
chunk_size = 10000  # Number of rows to hold in memory before writing to disk
is_first_chunk = True

# 2. Iterate through every single unique architecture graph in the database
for unique_hash in nasbench.hash_iterator():
    # Retrieve the architecture matrix and metrics
    fixed_stats, computed_stats = nasbench.get_metrics_from_hash(unique_hash)
    
    # Extract immutable features
    num_params = fixed_stats['trainable_parameters']
    ops_sequence = "-".join(fixed_stats['module_operations'])
    
    # 3. Flatten out the nested epochs and random trials
    for epoch in [4, 12, 36, 108]:
        for trial_idx, trial_data in enumerate(computed_stats[epoch]):
            
            # Construct a single flat row
            row = {
                'architecture_hash': unique_hash,
                'operations': ops_sequence,
                'trainable_parameters': num_params,
                'epoch': epoch,
                'trial': trial_idx + 1,
                'training_time_seconds': trial_data['final_training_time'],
                'train_accuracy': trial_data['final_train_accuracy'],
                'validation_accuracy': trial_data['final_validation_accuracy'],
                'test_accuracy': trial_data['final_test_accuracy']
            }
            buffer.append(row)
            
            # 4. Periodically flush to file to prevent RAM crashes
            if len(buffer) >= chunk_size:
                df = pd.DataFrame(buffer)
                df.to_csv(output_csv_path, mode='a', index=False, header=is_first_chunk)
                is_first_chunk = False
                buffer = []

# 5. Flush any remaining rows left in the buffer
if buffer:
    df = pd.DataFrame(buffer)
    df.to_csv(output_csv_path, mode='a', index=False, header=is_first_chunk)

print(f"Extraction complete! Saved to: {output_csv_path}")
