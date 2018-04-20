# Comp7404 Group Project (Group W)
## Emotion Detection (Group W)
** Emotion_Classifer_lite.py **
It is the simple CNN model program. You can input any image and label which you want.

For training Emotion_Classifer_lite.py
Please preapre the dataset file and the structure like that:
For dataset directoty
- dataset
  - training_set
    - label 1
      - (Training Image)
    - label 2
      - (Training Image)
    - ...
      - ...
  -  test_set
     - label 1
       - (Test Image)
     - label 2
       - (Test Image)
     - ...
       - ...
   
```
py Emotion_Classifer_lite.py t
```

For running Emotion_Classifer_lite.py to predict the iamge

```
py Emotion_Classifer_lite.py p /imagePath/tagetImage.png
```



This program is developed based on existing project.

Reference:
https://becominghuman.ai/building-an-image-classifier-using-deep-learning-in-python-totally-from-a-beginners-perspective-be8dbaf22dd8
> Venkatesh Tata

Reference:
https://github.com/oarriaga/face_classification
> B-IT-BOTS robotics team.
> Oarriaga

Reference
Training dataset Extract from IMDB



## Emotion Detection (Reference Project)
Step:
1. Use command line / terminal to go to the directory of ../face_classification/src

2. Enter command 
'''
python3 video_emotion_gender_demo.py {arg1} {arg2}
'''
- arg1 is the emtional model in directory ../face_classification/trained_model/emotion_model/{arg1}
- arg2 is the gender model in directory ../face_classification/trained_model/gender_model/{args}
3. If not enter the arg1 or arg2, it will use the default trained model for demo

