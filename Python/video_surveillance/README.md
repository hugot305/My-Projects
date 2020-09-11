# **Intelligent Video Surveillance with Deep Learning** 

---

**Intelligent Video Surveillance with Deep Learning**

In this deep learning project, my goal is to train an autoencoder for abnormal event detection. We train the autoencoder on normal videos. We identify the abnormal events based on the euclidean distance of the custom video feed and the frames predicted by the autoencoder

We set a threshold value for abnormal events. In this project, it is 0.00044 you can vary this threshold to experiment getting better results.

You can find the original paper [here](http://www.cse.cuhk.edu.hk/leojia/papers/abnormaldect_iccv13.pdf). 

---
### Dependencies
The only special libraries used in this project are: keras, tensorflow, numpy, and opencv.

You can install the dependencies like this:
```
    pip install keras
    pip install opencv
```

### Dataset
For this  project, there is a copy of the dataset used for training already included. 

### To run the project

```
    python train.py
    python test.py
```


