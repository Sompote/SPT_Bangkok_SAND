# Predicting Friction Angle of Bangkok Sand Using SPT

This project investigates the relationship between Standard Penetration Test (SPT) N-values and the friction angle of Bangkok Sand. It leverages a machine learning model, specifically a TensorFlow implementation, to establish a predictive framework. This repository provides the model architecture, pre-trained weights (if applicable), and potentially code for training, evaluation, and prediction.

![pp1](https://github.com/Sompote/SPT_Bangkok_SAND/assets/62241733/2e0d746c-8b94-49ac-a0fa-409b1e92a47f)

## Key Features

* Predicts Friction Angle: Estimates the friction angle of Bangkok Sand based on SPT N-values.
* Machine Learning Approach: Utilizes a TensorFlow model, likely a neural network architecture, for accurate and efficient predictions.
* Pre-trained Weights:Include pre-trained model weights for immediate use (if training data is unavailable).

![spt](https://github.com/Sompote/SPT_Bangkok_SAND/assets/62241733/705b7934-4ff0-4e98-9516-fbc173493407)



## Authors

- Dr. Sompote Youwai, AI Research Group, KMUTT


## Deployment

To deploy this project run
```
git clone https://github.com/Sompote/SPT_Bangkok_SAND
```



## Demo
```

import pickle
import numpy as np
import keras
def SPT_fee(stress=100, N=3):
  model1 = keras.models.load_model('model.h5')
  scaler_x = pickle.load(open('scaler.pkl', 'rb'))
  scaler_y = pickle.load(open('scaler_y.pkl', 'rb'))
  x = np.array([stress,N])
  x=np.reshape(x,(1,2))
  x_scale=scaler_x.transform(x)
  y_scale=model1.predict(x_scale)
  y_scale=np.reshape(y_scale,(1,1))
  y=scaler_y.inverse_transform(y_scale) #state parameter
  #calculate for friction angle
  fee=34.821-27.512*y
  return fee[0,0]
```

```
N =15

stress=363 #kPa
print('friction angle =',SPT_fee(stress=stress,N=N), 'degree')
```
```
1/1 [==============================] - 0s 148ms/step

friction angle = 33.91104 degree
```
