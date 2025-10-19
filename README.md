# Uni---Pattern-Recognition
# 🧠 Pattern Recognition – Assignments & Final Project

### Democritus University of Thrace  
**Department of Electrical and Computer Engineering**  
**Course:** Pattern Recognition (9th Semester)  
**Instructor:** Dr. I. Theodorakopoulos  

---

## 📘 Course Overview

The **Pattern Recognition** course introduces students to the fundamental principles and computational techniques used for **classification**, **clustering**, and **data analysis**.  
It combines statistical theory, optimization, and machine learning algorithms to teach how computers can learn to recognize patterns and make intelligent decisions.

Students learn to apply:
- **Bayesian decision theory** and parameter estimation methods.
- **Non-parametric techniques** such as Parzen Windows and k-Nearest Neighbors (k-NN).
- **Linear classifiers** like the Perceptron, Ho–Kashyap algorithm, and Linear Discriminants.
- **Support Vector Machines (SVMs)** for linear and nonlinear classification.
- **Clustering** and **dimensionality reduction** methods (e.g., PCA, Fisher discriminants).
- **Decision Trees**, **Random Forests**, and ensemble techniques.
- **Neural Networks** and **Deep Learning architectures** such as autoencoders.

The course emphasizes the implementation of these methods in **Python** and **MATLAB** on real datasets, bridging theory and practice.

---

## 💻 Repository Description

This repository contains my full set of implementations, reports, and analysis from the Pattern Recognition course.  
Each code project corresponds to a distinct stage in the progression from **classical statistical learning** to **modern deep learning**.  
The solutions are written in Python (and occasionally MATLAB) and structured to demonstrate understanding, experimentation, and visualization of results.

---

## 🧩 Implemented Methods and Applications

### 🔹 Statistical & Non-Parametric Classification
Implemented density estimation and Bayesian classification using:
- **Parzen window estimators** and **k-NN probability estimation** to approximate class-conditional densities.
- Visualization of probability distributions, decision regions, and the influence of parameters such as window width and neighborhood size.
- Comparison of **non-parametric** versus **geometric** classifiers in terms of generalization and smoothness.

### 🔹 Linear Classifiers & SVMs
Developed and compared several linear classification techniques:
- **Batch Perceptron** and **Ho–Kashyap algorithms** for separable 2D Gaussian datasets.
- **Support Vector Machines (SVMs)** for both linear and nonlinear separations using RBF and polynomial kernels.
- Hyperparameter tuning (e.g., regularization parameter `C`) with validation and cross-validation.
- Multiclass SVM classification using **one-vs-one** voting schemes.
- Evaluation through accuracy scores and confusion matrices.

### 🔹 Neural Networks and Deep Learning
Designed neural network models for structured data and image recognition:
- **Feedforward neural networks** trained with stochastic gradient descent, using both **Sigmoid** and **ReLU** activations.
- Monitoring of training and validation accuracy over epochs.
- **Precision–Recall** and **ROC** curve analysis, including AUC computation.

Developed **autoencoders** for the MNIST dataset:
- Built multi-layer encoders and decoders for image reconstruction and latent-space visualization.
- Examined the encoded feature distributions and generated new images from random latent points to explore the manifold of learned representations.

### 🔹 Decision Trees and Ensemble Learning
Explored tree-based learning and robustness to incomplete data:
- Implemented **Decision Tree** and **Random Forest** classifiers for the Breast Cancer Wisconsin dataset.
- Simulated different levels of **missing data** in both training and test sets.
- Analyzed the effect of data incompleteness on model accuracy and stability.
- Computed **feature importance** and compared model resilience between individual trees and ensemble methods.
- Evaluated classifier behavior using **Precision–Recall** curves, emphasizing medical decision-making reliability.

### 🔹 Applied Machine Learning Projects
Integrated multiple concepts in two real-world applications:

**1. Defect Detection in Additive Manufacturing (3D Printing)**  
Developed a classifier to detect process defects from **laser monitoring sensor signals**.  
The system distinguishes between *Normal operation* and two defect types (*A* and *B*), using optimized preprocessing, model selection, and validation to ensure robust signal classification.

**2. Molecular Binding Affinity Prediction**  
Designed an ML model to predict whether a **chemical molecule binds effectively** to a biological receptor.  
Used datasets with **continuous and binary molecular descriptors (3,473 features)**.  
Applied feature scaling, model tuning, and evaluation through **ROC–AUC** metrics.  
Generated submission-ready predictions for unseen test molecules.

---

## 🧠 Key Learning Outcomes

Through these projects, I developed a solid understanding of:
- The theoretical foundations of pattern recognition and statistical learning.
- The design, training, and evaluation of machine learning models.
- Practical implementation of **Bayesian**, **non-parametric**, **geometric**, and **neural** classifiers.
- Visualization of decision boundaries, feature spaces, and performance metrics.
- Application of ML techniques to **industrial**, **biomedical**, and **bioinformatics** problems.

---

## ⚙️ Technologies & Tools
- **Languages:** Python, MATLAB  
- **Libraries:** NumPy, pandas, scikit-learn, PyTorch, Matplotlib, seaborn  
- **Concepts:** Parzen Windows • k-NN • Perceptron • SVM • Autoencoders • Decision Trees • Random Forests • ROC & PR Curves  

---

## 📂 Repository Structure
