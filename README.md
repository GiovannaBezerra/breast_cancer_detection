<img src="https://user-images.githubusercontent.com/44107852/214916266-d4e61cb4-601f-4c55-b3c4-5d7430453662.png" align="right"
      width="90" height="90">
      
# BREAST CANCER DETECTION MODEL USING MACHINE LEARNING

In this project, a machine learning model was created to classify the prognosis of a breast cancer (malignant or benign) based on features computed from digitized images of of a fine needle aspirate (FNA) of a breast mass. This features describe characteristics of the cell nuclei present in the images.  


![density-plot](https://user-images.githubusercontent.com/44107852/214923267-de880202-a792-4e92-a8c3-8bfeb28fc305.jpg)

## Table Of Content  

[1. Problem understanding](#1problem-understanding)  
[2. How to use](#how-to-use)  
[3. Results](#results)  
[4. Notes and Considerations](#notes-and-considerations)  

## 1. Problem understanding  

Some asks could be answered in the present analysis:  
>   1. Which features are more influential in prognosis?  
>   2. Which machine learning model presents the highest accuracy in determining prognosis?  
>   3. What is the value of this accuracy? 

The data set used is **Breast Cancer Wisconsin (Diagnostic) Data Set** from Scikit-Learn, originally found on UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29.  

This dataset contains 569 instances and 30 attributes.  

Attribute Information:  
   * radius (mean of distances from center to points on the perimeter)
   * texture (standard deviation of gray-scale values)
   * perimeter
   * area
   * smoothness (local variation in radius lengths)
   * compactness (perimeter^2 / area - 1.0)
   * concavity (severity of concave portions of the contour)
   * concave points (number of concave portions of the contour)
   * symmetry
   * fractal dimension (“coastline approximation” - 1)  

The mean, standard error, and “worst” or largest (mean of the three worst/largest values) of these features were computed for each image, resulting in 30 features.

![describe](https://user-images.githubusercontent.com/44107852/214923411-df46787e-f20d-4a63-83a0-ca059a03a80b.jpg)


## 2. How to use  

### Installation and configuration 

```
# Clone this repository
git clone XXXXXXXXX

# Install development dependencies
pip install pandas
pip install numpy
pip instal matplotlib
pip install seaborn
pip install missingno
pip install -U scikit-learn

```

## 3. Results

With this work we can answer the questions presented initially.


![models](https://user-images.githubusercontent.com/44107852/214925767-701c71a8-2c3d-4777-b289-65de278263ee.jpg)


![model-result](https://user-images.githubusercontent.com/44107852/214923667-c69b8c2c-f0dc-4994-8469-99d3e681f911.jpg)


We only have **3** errors compared to **111** correct predictions which, together with **0.973684** of accuracy, should be a good quantification of model quality.

![confusion-matrix](https://user-images.githubusercontent.com/44107852/214923910-379651e1-fbb2-42f7-83c7-ac8cd3c82fa6.jpg)




## 4. Notes and Considerations  

During this development, I've learned how to use Dash and Plotly libraries, which was very challenging for me, in particular because of layout building and callbacks construction.

### References

<https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29>  
<https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset>  
<https://thecleverprogrammer.com/2022/03/08/breast-cancer-survival-prediction-with-machine-learning/>
