import pandas as pd
import os


def merge_csv_sets(first_set_path, second_set_path, output_dir):
    first_set_files = os.listdir(first_set_path)
    second_set_files = os.listdir(second_set_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, first_file in enumerate(first_set_files, start=1):
        for j, second_file in enumerate(second_set_files, start=1):
            first_df = pd.read_csv(os.path.join(first_set_path, first_file))
            second_df = pd.read_csv(os.path.join(second_set_path, second_file))
            min_lines = min(len(first_df), len(second_df))
            first_df = first_df.iloc[:min_lines, 1:]
            second_df = second_df.iloc[:min_lines, 1:]
            merged_df = pd.concat([first_df, second_df], axis=1)
            first_index = first_file.split("_")[1]
            output_filename = f"{first_index}_Left{i}_Right{j}.csv"
            output_path = os.path.join(output_dir, output_filename)

            merged_df.to_csv(output_path, index=False)

    for file in first_set_files:
        os.remove(os.path.join(first_set_path, file))
    for file in second_set_files:
        os.remove(os.path.join(second_set_path, file))

merge_csv_sets("left", "right", "output")