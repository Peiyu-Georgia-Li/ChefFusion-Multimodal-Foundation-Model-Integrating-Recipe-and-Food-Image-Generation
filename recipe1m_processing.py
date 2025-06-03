import json
import pandas as pd
from tqdm import tqdm
import csv
import re
import numpy as np

''' Remember to change the path !!!'''
with open('/home/path/to/layer1.json' , 'r') as file:
  data_1 = json.load(file)

with open('/home/path/to/layer2.json' , 'r') as file:
  data_2 = json.load(file)

###########################################################
#For layer1
id = []
title = []
part = []
url = []


def clean_line(line):
    '''
    Args:
        line: a string, such as food name, sentences...
    '''
    assert type(line) == str

    # all lowercase
    line = line.lower()

    # only reserve number and alphabets
    line = re.sub(r'[^a-z0-9+()/?!.,]', ' ', line)

    # replace things in brace
    line = re.sub(r'\([^)]*\)', '', line)

    # remove extra spaces
    line = re.sub(' +',' ',line).strip()
    return line

def get_list_of_strings(key , json_file):
    all_item_lists = []
    for  recipe in tqdm(json_file):
        items = recipe[key]
        item_list = []
        for line in items:
            cleaned = clean_line(line['text'])
            if cleaned:
                item_list.append(cleaned)

        all_item_lists.append(item_list)
    return all_item_lists

# Looping through and appending the values to a newly created list
for attr in tqdm(data_1):
  id.append(attr['id'])
  title.append(attr['title'])
  part.append(attr['partition'])
  url.append(attr['url'])

  # Building a dictionary with above values
data_1_dict = {'ID': id , 'food_title': title , 'partition': part , "recipe_url": url}

# Convert this beautiful dict into a fully fledge dataframe
data1_df = pd.DataFrame(data_1_dict)

# Using the above function and parsing out the values
ingredients_list = get_list_of_strings('ingredients' , data_1)
instructions_list = get_list_of_strings('instructions' , data_1)

data1_df['ingredients'] = pd.Series(ingredients_list)
data1_df['instructions'] = pd.Series(instructions_list)

###########################################################################################
# For layer2
id = []
for attr in tqdm(data_2):
  id.append(attr['id'])

data_2_dict = {'ID': id}
data2_df = pd.DataFrame(data_2_dict)

def get_list_of_strings(key , json_file):
    all_image_lists = []
    for re in tqdm(json_file):
        images = re[key]
        image_list = []
        for image in images:
            image_list.append(image['id'])
            break

        all_image_lists.append(image_list[0])
    return all_image_lists
# Using the above function and parsing out the values
images_list = get_list_of_strings('images' , data_2)

def get_list_of_string(key , json_file):
    all_image_lists = []
    for re in tqdm(json_file):
        images = re[key]
        image_list = []
        for image in images:
            image_list.append(image['url'])
            break

        all_image_lists.append(image_list[0])
    return all_image_lists
# Using the above function and parsing out the values
url_list = get_list_of_string('images' , data_2)

data2_df['images_id'] = pd.Series(images_list)
data2_df['images_url'] = pd.Series(url_list)

# merge dataframes based on ID column
merged_df = pd.merge(data1_df, data2_df, on = 'ID')

df0 = merged_df[['food_title','ingredients','instructions','images_id','partition']]


def filter(filename):

# reading tsv file
    dataFrame = pd.read_csv(filename,sep='\t')
    print("DataFrame...\n",dataFrame)

# select rows containing text "train"
    tr = dataFrame[dataFrame['partition'].str.contains('train')]
    tr.pop('partition')
    tr.to_csv('train_'+filename, sep="\t", index = False)

# select rows containing text "val"

    va= dataFrame[dataFrame['partition'].str.contains('val')]
    va.pop('partition')
    va.to_csv('val_'+filename, sep="\t", index = False)
# select rows containing text "test"
    te= dataFrame[dataFrame['partition'].str.contains('test')]
    te.pop('partition')
    te.to_csv('test_'+filename, sep="\t", index = False)

filter('ft_ing_ins_img_p.tsv')

# Concat train
df = pd.read_csv('./train_ft_ing_ins_img_p.tsv', delimiter='\t')
df['caption'] = 'This is food title: ' + df['food_title'] + ' This is ingredient: ' + df['ingredients'] + ' This is instruction: ' + df['instructions']
df.pop('instructions')
df.pop('ingredients')
df.pop('food_title')
df.rename(columns={"images_id": "image"}, inplace=True)
df.reset_index(drop=True, inplace=True)  # Reset index
df_train= df[['caption','image']]
df_train.to_csv('recipe1m_train.tsv', sep="\t", index=False)

#   Concat val
df = pd.read_csv('./val_ft_ing_ins_img_p.tsv', delimiter='\t')
df['caption'] = 'This is food title: ' + df['food_title'] + ' This is ingredient: ' + df['ingredients'] + ' This is instruction: ' + df['instructions']
df.pop('instructions')
df.pop('ingredients')
df.pop('food_title')
df.rename(columns={"images_id": "image"}, inplace=True)
df.reset_index(drop=True, inplace=True)  # Reset index
df_val = df[['caption','image']]
df_val.to_csv('recipe1m_val.tsv', sep="\t", index=False)



