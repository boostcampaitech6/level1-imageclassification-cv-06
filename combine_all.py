import csv
import os
import pandas as pd

csv_file_paths = ["./AGEOUTPUT.csv", "GENDEROUTPUT.csv", "MASKOUTPUT.csv"]

info_path = os.path.join("/data/ephemeral/home/data/eval", "info.csv")
info = pd.read_csv(info_path)
preds = [0 for _ in range(len(info))]

product = [1, 3, 6]
for idx, csv_file_path in enumerate(csv_file_paths):
    with open(csv_file_path, "r") as csv_file:
        # Create a CSV reader object
        csv_reader = csv.reader(csv_file)

        # Iterate over each row in the CSV file

        for _, row in enumerate(csv_reader):
            if _ == 0:
                continue
            # Each 'row' is a list representing a row in the CSV file
            print(int(row[1]))
            preds[_ - 1] += int(row[1]) * product[idx]

print(preds)
info["ans"] = preds
save_path = os.path.join("./", f"Sumup.csv")
info.to_csv(save_path, index=False)
