# Python-Topic-Modeling-using-Latent-Dirichlet-Allocation-LDA
Python Topic Modeling using Latent Dirichlet Allocation (LDA)

### importing libraries
```python
import random
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
```
The dataset we will use for the topic modeling is a very popular dataset on Kaggle. It contains 568,454 food reviews Amazon users left up to October 2012, we will divide the customer reviews into 5 groups using LDA.
Download the dataset [here](https://medium.com/r/?url=https%3A%2F%2Fwww.kaggle.com%2Fcode%2Fsdxingaijing%2Ftopic-model-lda-algorithm%2Fdata).

```python
data = pd.read_csv('amazon-reviews.csv')
data = data.head(2500)
data.head()
```
The dataset is quite voluminous so we will be using just 2,500 rows.

```python
data.shape
```


```python
data.isnull().sum()
```

```python
data.dropna()
```
Since the "Text" column contains the reviews, LDA will only be applied to that column; the other columns will be disregarded. Let's have a quick look at the "Text" column row 4.

```python
data['Text'][4]
```
The terms in our data must first be compiled into a vocabulary before we can use LDA, we can do this with the aid of a count vectorizer.

```python
vec = CountVectorizer(max_df=0.85, min_df=2, stop_words='english')
v_matrix = vec.fit_transform(data['Text'].values.astype('U'))

v_matrix
```

We will use LDA to generate topics and the probability distribution for each word in each topic's vocabulary.

```python
LDA = LatentDirichletAllocation(n_components=5, random_state=45)
LDA.fit(v_matrix)
```
Run a quick check to find words with the highest likelihood or probability for the first topic.

```python
topic_1 = LDA.components_[0]
```
Utilizing the argsort() method, we can sort the indexes in accordance with the probabilities. After sorting the 10 words with the greatest chances will then belong to the last 10 indexes of the array.

```python
top_topics = topic_1.argsort()[-10:]
top_topics
```
The value of the words may then be obtained from the vec object using these indexes.

```python
for i in top_topics:
    print(vec.get_feature_names()[i])
```
From the output, the words show that the first topic may be around food/pastries, let's print the words that have the highest probability for each of the five topics.

```python
for a,topic in enumerate(LDA.components_):
    print(f'Top 10 words for topic #{a}:')
    print([vec.get_feature_names()[a] for a in topic.argsort()[-10:]])
    print('\n')
```
The results show that the second topic often includes reviews about dog food, etc. The third topic could be reviews of online delivery services. You can notice that all of the categories share a few terms in common.

As the last step, we will include a column in the initial data frame. We may achieve this by sending our document-term matrix to the LDA.transform() method. In this way, the likelihood of each subject will be allocated to each document.

```python
new_topic_results = LDA.transform(v_matrix)
new_topic_results.shape
```

```python
data['Topic'] = new_topic_results.argmax(axis=1)
```

You should see (2500, 5), which indicates that there are 5 columns in each of the documents, each of which corresponds to a probability value for a different issue. Calling the argmax() function with the axis argument set to 1 will return the subject index with the highest value. Let's add a new column called "Topic" to the data frame that gives each row in the column a topic value.

```python
data.head()
```

 I will update this repository once I publish an article about this project.
 
 Happy coding!
