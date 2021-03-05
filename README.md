# Spatial Softmax and Soft Argmax
## In Tensorflow
Implementation of Spatial Softmax and Soft Argmax techniques used for keypoint localization using a Convolution Neural Network (CNN)

### Example
####  Testing Soft ArgMax layer
This layer has zero trainable parameter
```ruby
import numpy as np

import tensorflow as tf
from   tensorflow import keras
from   tensorflow.keras.models import Model

from   tensorflow.keras.layers import Conv2D, Input, Reshape
from CustomLayer import SpatialSoftmax, SoftArgMaxConv

def test_model(shape = (None, None, None), n_keys = 21):
    
    inputs = Input(shape = shape)
    x      = SoftArgMaxConv()(inputs)
    x      = Reshape((n_keys, -1))(x)
    
    return Model(inputs, x)
    
if __name__=='__main__':
    
    H = 25
    W = 25
    K = 3  # No. of keypoints
    
    model = test_model(shape = (H,W,K))
   
    data = np.zeros((H,W,K), dtype = np.float32)
    gt = []

    for c in range(K):
        x = np.random.randint(H)
        y = np.random.randint(W)
        
        data[x,y,c] = 1.
        gt.append([x,y])
        
    gt = np.array(gt)
    
    data  = np.expand_dims(data, axis=0)
    pred  = model.predict(data)[0]

```
