import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil

plt.style.use('ggplot')

# set variables
main_folder = './dataset/'
images_folder = main_folder + 'img_align_celeba/img_align_celeba/'
output_folder = 'splitted'

# import the data set that include the attribute for each picture
df = pd.read_csv(main_folder + 'list_attr_celeba.csv')

# Create the output folder and sub-folders
os.makedirs(os.path.join(output_folder, 'Male'), exist_ok=True)
os.makedirs(os.path.join(output_folder, 'Female'), exist_ok=True)

# Iterate through the CSV rows and move images to the appropriate sub-folder
for index, row in df.iterrows():
    image_id = row['image_id']
    male_attribute = row['Male']

    if male_attribute == 1:
        # Move to the "Male" folder
        source_path = os.path.join(images_folder, image_id)
        target_path = os.path.join(output_folder, 'Male', image_id)
        shutil.copyfile(source_path, target_path)
    elif male_attribute == -1:
        # Move to the "Female" folder
        source_path = os.path.join(images_folder, image_id)
        target_path = os.path.join(output_folder, 'Female', image_id)
        shutil.copyfile(source_path, target_path)
