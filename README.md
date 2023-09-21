# Pneumonia Detection using Transfer Learning with ResNet50
![ResNet50 Architecture](https://raw.githubusercontent.com/Aliraqimustafa/Pneumonia-Detection/main/imgs/1_sE2lzbGqffqXZmHeooB0-g.jpg)
Welcome to the "Pneumonia-Detection" repository. In this project, we provide code to help you utilize Transfer Learning with ResNet50 for the detection of pneumonia in medical scans. We achieved an accuracy of 97.8% and obtained the following confusion matrix:
[ 598, 10]
[41, 1693]


## Introduction
We employed Transfer Learning with the ResNet50 architecture to achieve these results. The power of Transfer Learning allows us to leverage pre-trained models and fine-tune them for our specific task.

## How ResNet50 Works
To understand the inner workings of ResNet50, refer to the following image:

![ResNet50 Architecture](https://raw.githubusercontent.com/Aliraqimustafa/Pneumonia-Detection/main/imgs/0_tH9evuOFqk8F41FG.png)

## How Transfer Learning Works
To grasp the concept of Transfer Learning, please refer to the following image:

![Transfer Learning](https://raw.githubusercontent.com/Aliraqimustafa/Pneumonia-Detection/main/imgs/1_9GTEzcO8KxxrfutmtsPs3Q.png)

## Requirements
To execute the code in this repository, you will need to install the following Python libraries:
- PyTorch
- Matplotlib
- Pandas
- Fastai

## Dataset
We used the following dataset for training and testing our model:
[Link to the dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## Using the Trained Model
You can use the final model directly on your computer by downloading the [export.pkl](https://drive.google.com/file/d/10Un0dDK_BaVQ6ainiBJUPr6yyNwkCb-l/view?usp=drive_link) file and then writing the following code:

```python
from fastai.learner import load_learner

learn_loaded = load_learner("/path/to/export.pkl")
```

## Making Predictions

After loading the model, you can make predictions using the following Python code:

```python
learn_loaded.predict(img)
```
## Contact

If you have any questions or need further assistance, feel free to reach out to me on the following platforms:

- Facebook: [Mustafa Mohammad](https://www.facebook.com/profile.php?id=100049592914479)
- Telegram: [Mustafa Mohammad](https://t.me/ha12qw)

## Sources

>For reference, you can check out the following source:
[pneumonia-detection-using-cnn-96-accuracy](https://www.kaggle.com/code/arbazkhan971/pneumonia-detection-using-cnn-96-accuracy/notebook)



Feel free to explore the code and use the model for your own projects. We encourage you to leverage the power of Transfer Learning with ResNet50 to tackle similar challenges or adapt it to your specific needs.
