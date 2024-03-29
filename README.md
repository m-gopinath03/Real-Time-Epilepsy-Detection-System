Epilepsy Monitoring System
Overview:
The Epilepsy Monitoring System is a project aimed at real-time monitoring and classification of human activities, particularly focusing on detecting seizures. It utilizes deep learning models to analyze live video streams of human activities and classify them as either normal or indicative of a seizure. The system provides timely alerts in case of a seizure detection, enabling prompt medical intervention.

Installation:
To install the necessary dependencies, follow these steps:

Clone the repository:
bash
git clone <repository_url>

Navigate to the project directory:
bash
cd epilepsy_monitoring_system

Install the required Python packages:
pip install -r requirements.txt

Usage
Data Pipeline: Run the data_pipeline.py script to preprocess video data, extract relevant features, balance the dataset, and prepare it for model training.

python data_pipeline.py

Model Development: Train and tune the deep learning model using the model_development.ipynb notebook. This notebook utilizes Keras Tuner to perform hyperparameter optimization and selects the best model for deployment.

Model Deployment: Deploy the trained model using the tkinter_deployment.ipynb notebook. This notebook contains the GUI application for real-time monitoring and classification of activities.

Project Structure
The project structure is organized as follows:

data/: Contains preprocessed CSV data files and raw video data(Absent).
notebooks/: Includes Jupyter notebooks for data processing, model development, evaluation, and deployment.
scripts/: Contains Python scripts for data preprocessing and other utility functions.
models/: Stores the trained model files.
README.md: Main documentation file.
requirements.txt: Lists all Python dependencies.
LICENSE: License information.
.gitignore: Specifies files and directories to be ignored by Git.
Model Architecture
The deep learning model architecture consists of LSTM layers followed by a custom attention layer for sequence processing. It also includes convolutional layers for feature extraction and dense layers for classification. Hyperparameter tuning is performed using Keras Tuner to optimize model performance.

Datasets
The project utilizes video datasets containing recordings of normal activities and seizure events. The datasets are preprocessed to extract pose, face, and hand landmarks using the MediaPipe library.

Evaluation Metrics
The model performance is evaluated based on accuracy and loss metrics during training and testing phases. Additionally, real-time monitoring accuracy is assessed through live video classification.

Examples and Tutorials
Example Notebook: Demonstrates usage of the data pipeline, model training, and deployment process.
License
This project is licensed under the MIT License.

Contributors
M Gopinath (@johndoe): Project Lead
Rahul Shukla (@janesmith): Data Preprocessing
Arya Sahu (@)

Contributions are welcome! Feel free to open issues or submit pull requests.

Acknowledgments
We thank the contributors to open-source libraries used in this project, including TensorFlow, Keras Tuner, MediaPipe, and others.

Contact
For questions or inquiries, please contact the project maintainers:

M Gopinath: mgopinath@iitrpr.ac.in
