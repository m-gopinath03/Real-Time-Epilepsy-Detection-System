# Epilepsy Detection System

## Overview
The Epilepsy Monitoring System is a project aimed at real-time monitoring and classification of human activities, particularly focusing on detecting seizures. It utilizes deep learning models to analyze live video streams of human activities and classify them as either normal or indicative of a seizure. The system provides timely alerts in case of a seizure detection, enabling prompt medical intervention.

## Table of Contents
- Installation
- Usage
- Project Structure
- Model Architecture
- Datasets
- Evaluation Metrics
- Examples and Tutorials
- License
- Contributors
- Acknowledgments
- Contact

## Installation
To install the necessary dependencies, follow these steps:
1. Clone the repository:
Clone the project repository by executing the following command in your terminal:

   git clone <https://github.com/m-gopinath03/Real-Time-Epilepsy-Detection-System-.git>
   
2. Navigate to the project directory:

   cd epilepsy_monitoring_system
   
3. Install the required Python packages:

   pip install -r requirements.txt   

## Usage

### Data Pipeline
Run the `data_pipeline.py` script to preprocess video data, extract relevant features, balance the dataset, and prepare it for model training.

To execute the data preprocessing pipeline, simply run the following commands in your terminal:
python data_pipeline.py


### Model Development
Train and tune the deep learning model using the `model_development.ipynb` notebook. This notebook utilizes Keras Tuner for hyperparameter optimization and selects the best model for deployment.

### Model Deployment
Deploy the trained model using the `tkinter_deployment.ipynb` notebook. This notebook includes the GUI application for real-time monitoring and classification of activities.

## Project Structure
The project structure is organized as follows:

- `data/`: Contains preprocessed CSV data files and raw video data (Absent).
- `notebooks/`: Includes Jupyter notebooks for data processing, model development, evaluation, and deployment.
- `scripts/`: Contains Python scripts for data preprocessing and other utility functions.
- `models/`: Stores the trained model files.
- `README.md`: Main documentation file.
- `requirements.txt`: Lists all Python dependencies.
- `LICENSE`: License information.
- `.gitignore`: Specifies files and directories to be ignored by Git.
  
## Model Architecture
The deep learning model architecture consists of LSTM layers followed by a custom attention layer for sequence processing. It also includes convolutional layers for feature extraction and dense layers for classification. Hyperparameter tuning is performed using Keras Tuner to optimize model performance.

## Datasets
To prepare the dataset for model training, a Makefile script is provided to automate the data preprocessing pipeline. This pipeline consists of three main steps:
- `Data Extraction`: Extracts relevant features from video data and saves them in CSV format.
- `Data Balancing`: Balances the dataset by augmenting the data to ensure equal representation of classes.
- `Data Labelling`: Labels the preprocessed data and saves it in a pickle format for easy access during model training.
To execute the data preprocessing pipeline, simply run the following commands in your terminal:
make extract 
make balance
make label
make preprocess this is for over all datapipeline
## Model Development

**To develop the deep learning model, follow these steps:**

1. Load the data from pickle files containing preprocessed features.
2. Define the model architecture including LSTM layers, convolutional layers, attention mechanism, and dense layers.
3. Utilize Keras Tuner to search for the best hyperparameters, such as the number of LSTM units, convolutional units, dense units, dropout rate, and learning rate.
4. Train the final model with the best hyperparameters selected from the search.
5. Evaluate the final model's performance on the test dataset to assess its accuracy and loss.


## Evaluation Metrics

During the development and testing of the model, several key evaluation metrics are used to assess its performance.

### Training History

To understand how the model's accuracy evolves over the training epochs, a visualization of the training history is created. This plot illustrates the training accuracy and validation accuracy over successive epochs, providing insights into the model's learning process.

### Model Evaluation

After training, the final model is evaluated using metrics like the ROC curve, AUC, and confusion matrix. The ROC curve shows the trade-off between sensitivity and specificity, while AUC quantifies overall performance. The confusion matrix summarizes model predictions compared to ground truth labels, providing insights into classification accuracy and error types. These metrics help assess model effectiveness and guide further improvements.

## Model Deployment

To deploy the epilepsy monitoring system using the trained deep learning model, follow these steps:

1. **Run the ClassifierApp**: Execute the `ClassifierApp` script to start the real-time monitoring and classification of activities. The application captures video input from the webcam, processes it using MediaPipe for landmark extraction, and feeds the extracted features into the trained model for classification.

2. **Activity Classification**: The application classifies activities into "Seizure" or "Normal" based on the detected poses and facial landmarks. The classification results are displayed in real-time on the screen, indicating the current activity along with the duration of the ongoing activity.

3. **Seizure Detection**: Upon detecting a seizure activity, the application initiates a timer to monitor the duration of the seizure episode. The duration is continuously updated and displayed on the screen until the seizure activity ceases.

4. **Termination**: To terminate the application, press the "q" key on the keyboard. This closes the application window and stops the video feed from the webcam.

Ensure that all dependencies, including OpenCV, MediaPipe, TensorFlow, and tkinter, are installed before running the application.

# License
This project is licensed under the MIT License.

## Contributors
M Gopinath : Project Lead

Rahul Shukla : Data Preprocessing

Contributions are welcome! Feel free to open issues or submit pull requests.

Acknowledgments
We thank the contributors to open-source libraries used in this project, including TensorFlow, Keras Tuner, MediaPipe, and others.

## Contact

If you have any questions, issues, or feedback regarding the Epilepsy Monitoring System, feel free to contact us:

- M Gopinath: mgopinath@iitrpr.ac.in

