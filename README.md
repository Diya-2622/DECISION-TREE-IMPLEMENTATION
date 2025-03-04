# DECISION-TREE-IMPLEMENTATION
*COMPANY NAME :* CODTECH IT SOLUTIONS

*NAME :* DIYA CHANDAN SINGH GEHLOT

*INTERN ID :* CT08RYD

*DOMAIN :* MACHINE LEARNING

*DURATION :* 4 WEEKS

*MENTOR :* NEELA SANTOSH KUMAR



### **Decision Tree Model: Overview**  
A **Decision Tree** is a supervised machine learning algorithm used for classification and regression tasks. It works by splitting the dataset into smaller subsets based on feature values, forming a tree-like structure. Each internal node represents a decision based on a feature, branches represent possible outcomes, and leaf nodes represent the final classification or prediction.  

Decision Trees are easy to interpret and visualize, making them popular for decision-making tasks. However, they are prone to overfitting, which can be controlled using pruning techniques, setting a maximum depth, or restricting the number of samples required to split a node.  

In this project, we implemented a **Decision Tree Classifier** using **Scikit-Learn** to classify the well-known **Iris dataset**, which contains features of different types of iris flowers.  

---

### **Tools Used**  

#### **1. Scikit-Learn (sklearn)**  
Scikit-Learn is a powerful Python library for machine learning that provides simple and efficient tools for data mining and analysis. It contains modules for classification, regression, clustering, dimensionality reduction, and model evaluation.  

In this project, we used Scikit-Learn for:  
- **Loading the dataset:** `datasets.load_iris()` loads the **Iris dataset**, which is a built-in dataset in Scikit-Learn containing features such as petal length, petal width, sepal length, and sepal width.  
- **Splitting the data:** `train_test_split()` splits the dataset into training and testing sets to evaluate the model’s performance.  
- **Building the Decision Tree Model:** `DecisionTreeClassifier()` initializes the decision tree model, with parameters like `criterion='gini'` (which uses the Gini index for splitting) and `max_depth=3` (which limits the depth of the tree to avoid overfitting).  
- **Model training:** `fit()` is used to train the model using the training data.  
- **Making predictions:** `predict()` is used to predict class labels on the test dataset.  
- **Evaluating the model:**  
  - `accuracy_score()` calculates the model’s accuracy by comparing the predicted labels with the actual labels.  
  - `classification_report()` provides a detailed breakdown of precision, recall, and F1-score for each class.  

#### **2. Matplotlib**  
Matplotlib is a Python visualization library used to plot graphs and visualize data. In this project, we used:  
- `plt.figure(figsize=(12, 8))` to set the figure size for the visualization.  
- `plot_tree()` to visualize the decision tree, displaying the feature names, class names, and the split conditions at each node.  

#### **3. NumPy**  
NumPy is a fundamental library for numerical computing in Python. It provides efficient handling of arrays and matrices. Although not heavily used in this project, NumPy plays a role in storing and manipulating the dataset and feature importance values.  

---

### **Application of Tools in the Project**  

1. **Dataset Handling:**  
   - The **Iris dataset** is loaded using `datasets.load_iris()`, providing four numerical features and three target classes (Setosa, Versicolor, and Virginica).  
   - Features are stored in `X`, and labels are stored in `y`.  

2. **Data Preprocessing and Splitting:**  
   - `train_test_split()` is used to divide the data into training (80%) and testing (20%) sets to ensure proper model evaluation.  

3. **Model Training:**  
   - A `DecisionTreeClassifier` is initialized with `criterion='gini'` and `max_depth=3` to prevent overfitting.  
   - The model is trained using `fit(X_train, y_train)`.  

4. **Model Evaluation:**  
   - Predictions are made on the test data using `predict(X_test)`.  
   - The accuracy score and classification report are generated to measure performance.  

5. **Visualization:**  
   - `plot_tree()` is used to generate a graphical representation of the decision tree.  
   - Each node shows the splitting condition, number of samples, and classification results.  

6. **Feature Importance Analysis:**  
   - The feature importance values (`feature_importances_`) indicate which features contribute the most to decision-making.  

---

### **Conclusion**  
This project demonstrated how to build, train, evaluate, and visualize a decision tree model using **Scikit-Learn**. The **Iris dataset** served as a simple yet effective dataset for classification. The decision tree structure was plotted using **Matplotlib**, while **Scikit-Learn** provided essential tools for machine learning tasks.  


#OUTPUT:

![Image](https://github.com/user-attachments/assets/749b7077-5f9f-4fba-b67d-6796b538384a)
