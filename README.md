# Python-Topic-Modeling-using-Latent-Dirichlet-Allocation-LDA
Python Topic Modeling using Latent Dirichlet Allocation (LDA)

```python
import random
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
```

```python
data = pd.read_csv('amazon-reviews.csv')
data = data.head(2500)
data.head()
```

```python
data.shape
```

```python
data.isnull().sum()
```

```python
data.dropna()
```

```python
data['Text'][4]
```

```python
vec = CountVectorizer(max_df=0.85, min_df=2, stop_words='english')
v_matrix = vec.fit_transform(data['Text'].values.astype('U'))

v_matrix
```

```python
LDA = LatentDirichletAllocation(n_components=5, random_state=45)
LDA.fit(v_matrix)
```

```python
topic_1 = LDA.components_[0]
```

```python
top_topics = topic_1.argsort()[-10:]
top_topics
```

```python
for i in top_topics:
    print(vec.get_feature_names()[i])
```

```python
for a,topic in enumerate(LDA.components_):
    print(f'Top 10 words for topic #{a}:')
    print([vec.get_feature_names()[a] for a in topic.argsort()[-10:]])
    print('\n')
```

```python
new_topic_results = LDA.transform(v_matrix)
new_topic_results.shape
```

```python
data['Topic'] = new_topic_results.argmax(axis=1)
```
