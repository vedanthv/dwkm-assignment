## Breast Cancer Diagnosis Using ML

<img src = "https://miro.medium.com/max/1400/1*pxFCmhRFTighUn88baLcSA.png">

### Problem
Given a dataset with geometrical features of the bresat(texture,concavity,perimeter etc...) we need to predict the accuracy of breast cancer in the patient

### Data
The dataset used for the demo is [Wisconsin Breast Cancer Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) on Kaggle.

### Objectives
1. Determine the features important to predict breast cancer.
2. Test performance using a range of classification models including SVM as mentionned in the question.

### Complete Pipeline
Follow along with the code form the ```breast-cancer-diagnosis.ipynb``` script

### Any questions?
Raise an issue [here]() and I'll get back to you!

### Best Solution to the Problem 

```
# To Setup the pipeline
steps = [('scaler', StandardScaler()),
         ('SVM', SVC())]

pipeline = Pipeline(steps)

# Specify the hyperparameter space: C is regularization strength while gamma controls the kernel coefficient. 
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}

# Create train & test sets

# Instantiate the GridSearchCV object: cv
cv =GridSearchCV(pipeline,parameters, cv=3)

# Fit to the training set
cv.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = cv.predict(X_test)

# Compute and print metrics
print("Tuned Model Parameters: {}".format(cv.best_params_))
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Best score is: {}".format(cv.best_score_))

ConfMatrix = confusion_matrix(y_test,cv.predict(X_test))
sns.heatmap(ConfMatrix,annot=True, cmap="Blues", fmt="d", 
            xticklabels = ['B', 'M'], yticklabels = ['B', 'M'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title("Confusion Matrix");
```

**Conclusion : Support Vector Machines are the best model for this problem with an accuracy of .95 and low FPR rate, a key indicator in medical diagnosis**

