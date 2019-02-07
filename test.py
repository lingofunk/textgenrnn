import sys, os
import pandas as pd

from senttextgenrnn.textgenrnn import textgenrnn

data_folder = '/media/student/8d1913cf-1155-47a5-a7db-b9a51f445d8f/student/data'

os.path.exists(data_folder)

os.listdir(data_folder)

reviews = pd.read_csv(
    data_folder + '/restaurant_reviews.csv',
    nrows=1,
    sep=',',
    index_col=0,
    usecols=['stars', 'text']
)

reviews.reset_index(inplace=True)

# print(reviews.head())
# print(reviews.head()['text'].values)

texts = reviews['text'].values
labels = reviews['stars'].values

model = textgenrnn()

###

word_level = False
new_model = False
num_epochs = 1
gen_epochs = 5
max_length = 40

###

model.train_on_texts(
    texts,
    context_labels=labels,
    word_level=word_level,
    num_epochs=num_epochs,
    gen_epochs=gen_epochs,
    max_length=max_length,
    new_model=new_model)

model.save()

print(model.generate(1))
print(model.generate(1, 1.0))
print(model.generate(1, 2.0))
print(model.generate(1, 3.0))
print(model.generate(1, 4.0))
print(model.generate(1, 5.0))
