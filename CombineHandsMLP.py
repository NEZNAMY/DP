import pandas as pd

data = pd.read_csv("input.csv")
grouped_data = data.groupby('ID')
combined_measurements = []
for patient_id, group in grouped_data:
    left_hand_measurements = group[group['Hand'] == 1].iloc[:, 2:].values
    right_hand_measurements = group[group['Hand'] == 2].iloc[:, 2:].values
    for left_measurement in left_hand_measurements:
        for right_measurement in right_hand_measurements:
            combined_measurement = list(left_measurement) + list(right_measurement)
            combined_measurements.append(combined_measurement)

feature_columns = data.columns[2:]
new_column_names = [f"{col} (Left)" for col in feature_columns] + [f"{col} (Right)" for col in feature_columns]
combined_df = pd.DataFrame(combined_measurements, columns=new_column_names)
combined_df.to_csv("output.csv", index=False)
