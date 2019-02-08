import sys, os
import pandas as pd

from textgenrnn.textgenrnn import textgenrnn

data_folder = '/media/student/8d1913cf-1155-47a5-a7db-b9a51f445d8f/student/data'

os.path.exists(data_folder)

os.listdir(data_folder)

reviews = pd.read_csv(
    data_folder + '/restaurant_reviews.csv',
    nrows=100,
    sep=',',
    index_col=0,
    usecols=['stars', 'text']
)

reviews.reset_index(inplace=True)

# print(reviews.head())
# print(reviews.head()['text'].values)

texts = reviews['text'].values
labels = reviews['stars'].values

label2sentiment = {
    1.0: -1,
    2.0: -0.8,
    3.0: 0,
    4.0: +0.8,
    5.0: +1
}

sentiments = [label2sentiment[label] for label in labels]

# print(sentiments)

model = textgenrnn()

###

word_level = False
new_model = False
num_epochs = 2
gen_epochs = 1
max_length = 40

###

# model.train_on_texts(
#     texts,
#     context_labels=labels,
#     word_level=word_level,
#     num_epochs=num_epochs,
#     gen_epochs=gen_epochs,
#     max_length=max_length,
#     new_model=new_model)
#
# model.save()
#
# print('Samples')
#
# sentiment_values = [-1, -0.8, -0.5, -0.1, 0, +0.1, +0.5, +0.8, +1]
#
# for sentiment_value in sentiment_values:
#     print('Sentiment:', sentiment_value)
#     print(model.generate(1, sentiment_value, return_as_list=True)[0])
#     print()
