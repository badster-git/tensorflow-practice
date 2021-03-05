### The data used in this tutorial are taken from the Titanic passenger list. The model will predict the likelihood a passenger survived based on characteristics like age, gender, ticket class, and 
###	whether the person was traveling alone.
from __future__ import absolute_import, division, print_function, unicode_literals
import functools

import pandas as pd
import numpy as np
import tensorflow as tf

TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"


train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

df = pd.read_csv(train_file_path)


df['survived'] = pd.Categorical(df['survived'])
df['survived'] = df.survived.cat.codes
print(df.head())


survived = df.pop('survived')
dataset = tf.data.Dataset.from_tensor_slices((df.values, survived.values))




#def get_dataset(file_path, **kwargs):
#  dataset = tf.data.experimental.make_csv_dataset(
#      file_path,
#      batch_size=5, # Artificially small to make examples easier to show.
#      label_name=LABEL_COLUMN,
#      na_value="?",
#      num_epochs=1,
#      ignore_errors=True, 
#      **kwargs)
#  return dataset
#
#raw_train_data = get_dataset(train_file_path)
#raw_test_data = get_dataset(test_file_path)