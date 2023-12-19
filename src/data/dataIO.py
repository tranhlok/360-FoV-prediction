'''
DataIO for the datset
'''

import os
import pandas as pd
import numpy as np
import os
import shutil

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

def split_act(directory,act_dir, activities):
    # Loop through each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            # Extract the activity name from the file name
            for activity in activities:
                if activity in filename.lower():
                    activity_name = activity
                    break
            else:
                print(f"Activity not found in file name: {filename}")
                continue

            # Create a folder for this activity if it doesn't exist
            activity_folder = os.path.join(act_dir, activity_name)
            if not os.path.exists(activity_folder):
                os.makedirs(activity_folder)

            # Move the file to the appropriate folder
            shutil.copy(os.path.join(directory, filename), os.path.join(activity_folder, filename))

    print("Files have been sorted into folders based on activities.")


# Set the directory where your CSV files are stored
raw_dir = os.path.expanduser("~/360-FoV-prediction/data/raw")
processed_dir = os.path.expanduser("~/360-FoV-prediction/data/processed")
act_dir = os.path.expanduser("~/360-FoV-prediction/data/processed_by_activity")
# Define the list of activities
activities = ["chatting", "cleaning_whiteboard", "news_interviewing", "pulling_trolley", "presenting", "sweep"]

convert_text_to_csv(raw_dir, processed_dir)
split_act(processed_dir, act_dir, activities)
print("FINISHED, PROCEED TO TRAIN MODEL.")