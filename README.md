# Image-Classification---Machine-Learning-Project

## Handwritten Digit Recognition with Machine Learning

This project focuses on classifying handwritten digits (0 to 9) using machine learning models built with the Scikit-Learn library and deploying the best-performing model through the Streamlit library. Using the MNIST dataset, we preprocess the data, train various machine learning models (Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbors), and determine the best model based on performance metrics. The Random Forest Classifier emerged as the most accurate model and was deployed in a user-friendly web application for real-time digit recognition.

The application allows users to upload an image of a handwritten digit, which is preprocessed and passed through the trained model to predict the digit accurately. This showcases a complete pipeline, from dataset exploration to machine learning model deployment.

---

## Features

- **Dataset Handling**: Utilizes the MNIST dataset for training and testing the models.
- **Data Preprocessing**:
  - Standardization for consistent scaling.
  - Principal Component Analysis (PCA) for dimensionality reduction, retaining 95% variance.
- **Model Training and Selection**:
  - Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbors models implemented.
  - Hyperparameter tuning using GridSearchCV for optimized performance.
- **Evaluation**:
  - Confusion matrix and accuracy scores for validation and testing.
  - Random Forest Classifier selected as the best-performing model.
- **Streamlit Deployment**:
  - User-friendly interface for uploading and processing handwritten digit images.
  - Real-time predictions with visual feedback.

---

## Technology Stack

- **Programming Language**: Python
- **Libraries and Tools**:
  - **Machine Learning**: Scikit-Learn
  - **Data Handling**: NumPy, Pandas
  - **Visualization**: Matplotlib
  - **Deployment**: Streamlit
  - **Utilities**: Joblib for model serialization, PIL for image processing

--- 
## Conclusion

This project demonstrates a comprehensive machine learning workflow, from data preprocessing and model development to deployment. The Random Forest Classifier, with its robust accuracy, was chosen as the final model and integrated into a Streamlit application. The outcome is a fully functional tool that accurately predicts handwritten digits from user-uploaded images, highlighting the capabilities of machine learning and its real-world applicability.

---
