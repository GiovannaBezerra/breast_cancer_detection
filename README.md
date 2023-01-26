<img src="https://user-images.githubusercontent.com/44107852/214916266-d4e61cb4-601f-4c55-b3c4-5d7430453662.png" align="right"
      width="90" height="90">
      
# BREAST CANCER DETECTION MODEL USING MACHINE LEARNING

In this project, a machine learning model was created to classify the prognosis of a breast cancer (malignant or benign) based on features computed from digitized images of of a fine needle aspirate (FNA) of a breast mass. This features describe characteristics of the cell nuclei present in the images.  

![density-plot](https://user-images.githubusercontent.com/44107852/214923267-de880202-a792-4e92-a8c3-8bfeb28fc305.jpg)

## Table Of Content  

- [Goals and objectives](#goals-and-objectives)  
- [How to use](#how-to-use)  
- [Project steps](#project-steps)  
- [Results](#results)  
- [Notes and Considerations](#notes-and-considerations)  

## Goals and objectives  

The intention of this project is to answer follow questions: 
>   1. Which features are more influential in prognosis?  
>   2. Which machine learning model presents the highest accuracy in determining prognosis?  
>   3. What is the value of this accuracy? 

The data set used is **Breast Cancer Wisconsin (Diagnostic) Data Set** from Scikit-Learn, originally found on UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29.  

This dataset contains 569 instances and 30 attributes.  

![describe](https://user-images.githubusercontent.com/44107852/214923411-df46787e-f20d-4a63-83a0-ca059a03a80b.jpg)

## How to use  

### Installation and configuration 

```
# Clone this repository
git clone https://github.com/GiovannaBezerra/breast_cancer_detection.git

# Install development dependencies
pip install pandas
pip install numpy
pip instal matplotlib
pip install seaborn
pip install missingno
pip install -U scikit-learn
```

## Project steps  

1. Problem understanding
2. Load dataset
3. Data exploration
4. Data Pre-Processing (missing value treatment / train and test data split)
5. Classification Models
6. Model Result

## Results    

Five models were evaluated for performance for three dataset configurations (original, standardized, and normalized):
- Logistic Regression
- KNN
- Decision Tree
- Naive Bayes
- Support Vector Machine

![model-result](https://user-images.githubusercontent.com/44107852/214923667-c69b8c2c-f0dc-4994-8469-99d3e681f911.jpg)

The results suggest that better model is ScaledLR, it means **Logistic Regression for Standardized dataset**, wich accuracy mean reaches **0.958261**

![models](https://user-images.githubusercontent.com/44107852/214925767-701c71a8-2c3d-4777-b289-65de278263ee.jpg)

Then, to verify the model performance, the model was tested into test dataset.

```
# Accuracy approximation in the test dataset:
rescaledTestX = scaler.transform(X_test)
predictions = model.predict(rescaledTestX)

print("Test accuracy: %f" % accuracy_score(Y_test, predictions))
```
Test accuracy: 0.973684

Finally the confusion matrix show us a comparative performance between true labels and predicted labels.

![confusion-matrix](https://user-images.githubusercontent.com/44107852/214923910-379651e1-fbb2-42f7-83c7-ac8cd3c82fa6.jpg)

Only **3** errors compared to **111** correct predictions which, together with **0.973684** of accuracy, should be a good quantification of model quality.

## Notes and Considerations  

With this work was possible to answer the questions presented initially. During this development, I learned how to structure a classification model building project, going through data exploration and pre-processing until model evaluation and results.

### References

<https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29>  
<https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset>  
<https://thecleverprogrammer.com/2022/03/08/breast-cancer-survival-prediction-with-machine-learning/>
