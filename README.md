# Classroom-Engagement-Detection-using-Facial-Expression-Recognition
Student engagement is a key concept in education. Facial expression of individuals can be utilized as an indicator to detect their engagement level. This project can recognize facial expression from the input video or webcam using **Convolutional Neural Network**.  
**FER-2013** dataset from Kaggle is used in training the Deep Learning model. Human accuracy on this FER-2013 dataset is **+-65%**. Performance evaluation was done using average accuracy on seven classes from the confusion matrix based on the original splitting of dataset.  
The highest classification accuracy was **70.44%** using ensemble of five deep CNN based on hard voting, followed by **68.99%** using a shallow CNN. Some techniques such as haarcascade face detector, data augmentation, mix-max normalization, histogram equalization were used.   
A simple web app is built to demonstrate the expression-based engagement detector using **Flask** as backend and **SQLite** as database to store the results.

Convolutional Neural Network is faster than Ensemble Model.