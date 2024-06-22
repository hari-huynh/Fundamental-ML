# Mini-Project for Fundamentals of Machine Learning Course
![background](./materials/ai_wp.jpg)
This repository contains the code and data for a mini-project on facial expression recognition using machine learning algorithms.


## 📑 Project Policy
- Team: group should consist of 3-4 students.

    |No.| Student Name    | Student ID |
    |:--------:|:--------:|:-------:|
    |1|Nguyễn Quang Trường |2110429|
    |2|Huỳnh Long Hải|21110286|
    |3|||
    |4|||

- The submission deadline is strict: **11:59 PM** on **June 22nd, 2024**. Commits pushed after this deadline will not be considered.


## <img src="https://github.com/hari-huynh/Fundamental-ML/assets/139192880/f9594288-b706-432f-8f9d-4bd09923325e" alt="NVIDIA GPU" style="width:40px;height:40px;"> cuML 


![cuML Logo](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQI8LeBSFHiHGsH4zXDeJggDC_FsUf6KpWKMg&s)
![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTp1XHEZzc2yZdVlFiPYvC305i59Zq5kHxiiA&s)

This project utilizes cuML for GPU-accelerated machine learning. cuML is part of RAPIDS AI, offering high-performance algorithms that speed up training and inference.

  -  🚀 Speed
cuML leverages GPUs to accelerate processing, significantly faster than CPUs, especially with large datasets.

 -  📊 High Performance
Ideal for large datasets due to GPU's parallel processing capabilities, reducing processing time and improving efficiency.

 - 🔧 Easy Integration
cuML's API is similar to scikit-learn, making it easy to transition and integrate into existing workflows.

 - 📚 Multiple Algorithms
Supports a wide range of algorithms: regression, classification, clustering, PCA, and more.

 - 🌐 Community & Documentation
Rich documentation and a large community provide ample support and learning resources.
[Document](https://docs.rapids.ai/api/cuml/stable/)

##
<img src="https://raw.githubusercontent.com/optuna/optuna/master/docs/image/optuna-logo.png" alt="" style="width:200px;height:30px;">

This project use Optuna for hyperparameter optimization. Optuna is an automatic hyperparameter optimization framework designed to improve the performance of machine learning models. Optuna uses advanced algorithms to efficiently search for the best hyperparameters, enhancing model performance.







## 📦 Project Structure

The repository is organized into the following directories:

- **/data**: This directory contains the facial expression dataset. You'll need to download the dataset and place it here before running the notebooks. (Download link provided below)
- **/notebooks**: This directory contains the Jupyter notebook ```EDA.ipynb```. This notebook guides you through exploratory data analysis (EDA) and classification tasks.

## ⚙️ Usage

This project is designed to be completed in the following steps:

1. **Fork the Project**: Click on the ```Fork``` button on the top right corner of this repository, this will create a copy of the repository in your own GitHub account. Complete the table at the top by entering your team member names.

2. **Download the Dataset**: Download the facial expression dataset from the following [link](https://mega.nz/file/foM2wDaa#GPGyspdUB2WV-fATL-ZvYj3i4FqgbVKyct413gxg3rE) and place it in the **/data** directory:

3. **Complete the Tasks**: Open the ```notebooks/EDA.ipynb``` notebook in your Jupyter Notebook environment. The notebook is designed to guide you through various tasks, including:
    
    1. Prerequisite
    2. Principle Component Analysis
    3. Image Classification
    4. Evaluating Classification Performance 

    Make sure to run all the code cells in the ```EDA.ipynb``` notebook and ensure they produce output before committing and pushing your changes.

5. **Commit and Push Your Changes**: Once you've completed the tasks outlined in the notebook, commit your changes to your local repository and push them to your forked repository on GitHub.


Feel free to modify and extend the notebook to explore further aspects of the data and experiment with different algorithms. Good luck.
