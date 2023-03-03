# etiology_classifier


## Classifies unknown etiologies in audiology dataset using Naive Bayes

* For many patients in an audiology clinic, it is unclear what disease or condition causes their hearing loss. This is apparent in the Audiology (Standardized) Data Set, available from: https://archive.ics.uci.edu/ml/datasets/Audiology+%28Standardized%29

![](Figures/Figure1.png) 

* This code trains a model on patients with known etiologies with 80% accuracy, which is then applied to patients with unknown etiologies.

![](Figures/Figure2.png)

* Feature selection is calculated to determine the most important predictors of etiology.
![](Figures/Figure3.png)