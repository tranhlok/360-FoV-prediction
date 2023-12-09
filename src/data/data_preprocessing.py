import os
import csv
import pandas as pd
import numpy as np

def convert_text_to_csv(input_folder, output_folder):
      # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all text files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + ".csv"
            output_path = os.path.join(output_folder, output_filename)

            # Read the text file and handle the first line
            data = []
            # first line sep by ",", data points are sep by tab space
            with open(input_path, "r", encoding='utf-8') as fr:
                columns = fr.readline().strip().split(', ')
                for line in fr.readlines(): #[1:] here if dont want first frames
                    data.append(list(map(lambda x: float(x), line.split(' '))))
            data = np.array(data)
            print(output_filename)
            print(data.shape)
            # filter noise data
            if data.shape[1] >= 28:
              df = pd.DataFrame(data, columns=columns)
              df.set_index(df.columns[0], inplace=True)
              # df.drop_duplicates(inplace=True)

              df.to_csv(output_path, sep=',')



convert_text_to_csv(os.path.expanduser("~/360-FoV-prediction/data/user_movement"), os.path.expanduser("~/360-FoV-prediction/data/processed"))