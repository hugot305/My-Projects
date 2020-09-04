# **Speech Emotion Recognition** 

---

**Speech Emotion Recognition**

In this project my goal is to build a model to recognize emotion from speech using the librosa and sklearn libraries and the RAVDESS dataset.

We'll use the libraries librosa, soundfile, and sklearn (among others) to build a model using an MLPClassifier. This model will be able to recognize emotion from sound files. We will load the data, extract features from it, then split the dataset into training and testing sets. Then, we’ll initialize an MLPClassifier and train the model. Finally, we’ll calculate the accuracy of our model.

---
### Dependencies
The only special libraries used in this project are: librosa, soundfile, numpy, sklearn and pyaudio.

You can install the dependencies like this:
```
    pip install librosa
    pip install soundfile
    pip install numpy
    pip install scikit-learn
    pip install pyaudio
```

### Dataset
For this  project, we’ll use the RAVDESS dataset; this is the Ryerson Audio-Visual Database of Emotional Speech and Song dataset, and is free to download. This dataset has 7356 files rated by 247 individuals 10 times on emotional validity, intensity, and genuineness. The entire dataset is 24.8GB from 24 actors, but we’ve lowered the sample rate on all the files, and you can downloaded from [here](https://drive.google.com/file/d/1wWsrN2Ep7x6lWqOXfr4rpKGYrJhWc8z7/view) 
### To run the project

```
    python speech.py
```

## Output

    Features extracted: 180
    Accuracy: 75.97%


