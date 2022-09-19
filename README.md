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
