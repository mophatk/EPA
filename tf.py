import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

bank = pd.read_csv('bank.csv', sep=";")
print(bank['contact'].unique())
print(bank['job'].unique())
print(bank['education'].unique())
print(bank['default'].unique())
print(bank['loan'].unique())
print(bank['month'].unique())
print(bank['day_of_week'].unique())
print(bank['poutcome'].unique())



bank['target'] = np.where(bank['y']=='no', 0, 1)
#print(bank.info())

train, test = train_test_split(bank, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)

print(len(train),'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

def df_to_dataset(bank, shuffle=True, batch_size=128):
    bank = bank.copy()
    labels = bank.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(bank), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(bank))
    ds = ds.batch(batch_size)
    return ds

batch_size = 20
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

#for feature_batch, label_batch in train_ds.take(1):
#  print('Every feature:', list(feature_batch.keys()))
#  print('A batch of ages:', feature_batch['age'])
#  print('A batch of targets:', label_batch )

#We will use this batch to demonstrate several types of feature_columns
example_batch = next(iter(train_ds))[0]

## A utility method to create a feature column
# and to transform a batch of data
def demo(feature_column):
  feature_layer = layers.DenseFeatures(feature_column)
#  print(feature_layer(example_batch).numpy())

camp = feature_column.numeric_column('campaign')
#demo(camp)

dura = feature_column.numeric_column('duration')
#demo(dura)

age = feature_column.numeric_column('age')
age_buckets = feature_column.bucketized_column(age, boundaries=[1, 3, 5])
#demo(age_buckets)

marital = feature_column.categorical_column_with_vocabulary_list(
 'marital', ['married', 'single', 'divorced']
)
marital_one_hot = feature_column.indicator_column(marital)
#demo(marital_one_hot)

feature_columns = []

#numeric cols
for header in ['age', 'duration', 'campaign', 'pdays', 'previous', ]:
    feature_columns.append(feature_column.numeric_column(header))

#bucketized cols
age = feature_column.numeric_column('age')
age_buckets = feature_column.bucketized_column(age, boundaries=[1, 2, 3, 4, 5])
feature_columns.append(age_buckets)

#indicator_columns
indicator_column_names = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
for col_name in indicator_column_names:
    categorical_column = feature_column.categorical_column_with_vocabulary_list(
        col_name, bank[col_name].unique()
    )
    indicator_column = feature_column.indicator_column(categorical_column)
    feature_columns.append(indicator_column)

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

batch_size = 128
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dropout(.1),
    layers.Dense(1)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_ds,
          validation_data=val_ds,
          epochs=10)

loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)
