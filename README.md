# **Disease Classification Using Machine Learning**

This repository contains the implementation of a project focused on classifying diseases based on symptoms using machine learning models. The project involves multiple iterations of dataset preprocessing, implementation of models, and visualization of results.

---

## **Repository Structure**

### 1. **Datasets**
Contains the original dataset and processed datasets across different iterations. Each iteration folder includes:
- `*.csv` files: The dataset at that iteration stage.
- `*_Cleaning_Code.py` files: Scripts used to clean and preprocess the dataset in that iteration.

**Subfolders**:
- **Original Dataset**: Raw dataset used as the starting point for preprocessing.
- **First Iteration** to **Seventh Iteration**: Subsequent iterations of dataset cleaning and feature engineering.

### 2. **Implementation**
Contains the Jupyter Notebook used to train and evaluate machine learning models:
- `Implementation.ipynb`: The main notebook for preprocessing, training, and testing machine learning models (Decision Trees, SVM, and KNN).

### 3. **Visualization**
Contains code for visualizing results:
- `Line_Graph_Code.ipynb`: Jupyter Notebook for generating line graphs and other visualizations to analyze model performance.

---

## **Models Used**
Three machine learning models were implemented:
1. **Decision Trees**: Best for interpretability and training accuracy.
2. **Support Vector Machines (SVM)**: Achieved the highest testing accuracy.
3. **K-Nearest Neighbors (KNN)**: Performed moderately across both training and testing phases.

---

## **Key Features**
- **Dataset Preprocessing**: Multi-label binarization for symptoms, handling missing values, and creating binary features for symptoms.
- **Evaluation Metrics**: Models were evaluated using accuracy, confusion matrix, and F1 score.
- **Iterations**: Multiple iterations were performed to refine the dataset and improve model performance.

---

## **How to Use**
1. **Explore Datasets**:
   - Navigate to the `Datasets` folder to review the datasets at various preprocessing stages.
   - Use the cleaning scripts for understanding or replicating data preprocessing.

2. **Run the Implementation**:
   - Open `Implementation.ipynb` in a Jupyter environment.
   - Execute the cells step-by-step to preprocess data, train models, and evaluate performance.

3. **Visualize Results**:
   - Open `Line_Graph_Code.ipynb` to generate performance visualizations.

---

## **Project Goals**
- Classify 40 diseases based on symptoms using machine learning models.
- Compare the performance of Decision Trees, SVM, and KNN.
- Identify the most relevant symptoms for disease prediction.

---

## **Future Work**
- Explore advanced models like deep learning to improve classification performance.
- Address overlapping symptoms for better differentiation of diseases.

---

## **Acknowledgements**
This project uses the **SymptomsDisease246k dataset** from [Hugging Face](https://huggingface.co/datasets/fhai50032/SymptomsDisease246k).

---

## **License**

This project is licensed under the [MIT License](./LICENSE).  
You are free to use, modify, and distribute this project, provided proper credit is given to the original authors. See the LICENSE file for more details.

---
