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
