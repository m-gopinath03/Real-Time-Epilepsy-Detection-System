# Real-Time Epilepsy Detection System 🧠⚡
A comprehensive deep learning system for real-time epilepsy seizure detection using computer vision and human pose estimation. The system analyzes live video streams to classify human activities as normal or indicative of seizures, providing timely medical alerts.
## 🌟 Features
### Advanced Computer Vision
- **MediaPipe Integration**: Holistic landmark extraction for pose, face, and hand tracking
- **Real-Time Processing**: Live video stream analysis with 25 FPS processing
- **Multi-Modal Detection**: Combines pose landmarks, facial expressions, and hand movements
- **Robust Tracking**: High confidence detection with missing landmark compensation
### Deep Learning Architecture
- **LSTM Networks**: Sequential pattern recognition for temporal activity analysis
- **CNN Integration**: Spatial feature extraction with 1D convolutions
- **Attention Mechanism**: Custom attention layer for feature prioritization
- **Hyperparameter Optimization**: Keras Tuner for automated model tuning
### Data Processing & Balancing
- **SMOTE Implementation**: Synthetic minority oversampling for dataset balancing
- **Automated Pipeline**: Makefile-driven preprocessing with configurable parameters
- **Feature Engineering**: 321-dimensional feature vectors from landmark coordinates
- **Temporal Segmentation**: 75-frame sliding windows for sequence analysis
### Real-Time Deployment
- **Tkinter GUI**: User-friendly interface for live monitoring
- **Seizure Timer**: Duration tracking for epileptic episodes
- **Visual Feedback**: Real-time activity classification display
- **Alert System**: Immediate notifications for seizure detection
## 🎬 Demo
### Real-Time Detection Interface
- Live webcam feed with landmark visualization
- Real-time seizure/normal classification
- Seizure duration timing and monitoring
- Visual pose estimation overlays
  
## 🏗️ Architecture
[20]
```
Real-Time-Epilepsy-Detection-System/
├── notebooks/                    # Jupyter notebooks for development
│   ├── data_extraction.ipynb    # Video processing & landmark extraction
│   ├── data_labelling_balancing.ipynb  # SMOTE balancing & segmentation
│   ├── model_development.ipynb  # LSTM-CNN model with attention
│   ├── model_evaluation.ipynb   # Performance metrics & visualization
│   └── tkinter_deploying.ipynb  # Real-time GUI deployment
├── scripts/                     # Automation scripts
│   ├── data_pipeline.py        # Complete preprocessing pipeline
│   └── Makefile                # Build automation for data processing
├── requirements.txt            # Python dependencies
└── README.md                  # Project documentation
```
## 🚀 Quick Start
### Prerequisites
- **Python 3.8+**
- **Webcam** (for real-time detection)
- **GPU Support** (recommended for training)
### Installation
1. **Clone Repository**:
   ```bash
   git clone https://github.com/m-gopinath03/Real-Time-Epilepsy-Detection-System.git
   cd Real-Time-Epilepsy-Detection-System
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare Data Structure**:
   ```bash
   mkdir -p data/{videos/{seizure,normal},CSV/{seizure,normal},pickle}
   ```
4. **Run Data Pipeline**:
   ```bash
   cd scripts
   make preprocess  # Complete pipeline
   # OR individual steps:
   make extract    # Video processing & landmark extraction
   make balance    # Dataset balancing with SMOTE
   make label      # Temporal segmentation & labeling
   ```
5. **Train Model**:
   ```bash
   jupyter notebook notebooks/model_development.ipynb
   ```
6. **Deploy Real-Time System**:
   ```bash
   jupyter notebook notebooks/tkinter_deploying.ipynb
   ```
### First Steps
1. **Data Preparation**: Place seizure and normal videos in respective folders
2. **Feature Extraction**: Run landmark extraction on video datasets
3. **Model Training**: Train LSTM-CNN model with hyperparameter tuning
4. **Real-Time Testing**: Launch GUI application for live detection
## 🛠️ Tech Stack
| Category | Technology | Purpose |
|----------|------------|---------|
| **Computer Vision** | MediaPipe 0.8.7 | Holistic landmark detection |
| **Deep Learning** | TensorFlow 2.7.0 | Model training & inference |
| **Video Processing** | OpenCV 4.5.3 | Real-time video capture |
| **Data Processing** | Pandas 1.3.3 | Data manipulation |
| **Model Tuning** | Keras Tuner 1.0.4 | Hyperparameter optimization |
| **Data Balancing** | Imbalanced-learn 0.8.1 | SMOTE implementation |
| **GUI Framework** | Tkinter | Real-time interface |
| **Video Editing** | MoviePy 1.0.3 | Video preprocessing |
| **Build System** | Makefile | Pipeline automation |

## 📊 Model Evaluation
### Performance Metrics
- **Accuracy**: Training and validation accuracy tracking
- **ROC Curve**: True positive vs false positive rate analysis
- **AUC Score**: Area under curve for model performance
- **Confusion Matrix**: Detailed classification results
- **Precision/Recall**: Seizure detection sensitivity

## 🖥️ Real-Time Deployment
### GUI Application Features
- **Live Video Feed**: Real-time webcam processing
- **Landmark Visualization**: MediaPipe pose estimation overlay
- **Classification Display**: Normal/Seizure status indication
- **Timer Functionality**: Seizure duration tracking
- **Alert System**: Visual notifications for detections

## 📈 Key Features Comparison
| Feature | Training Phase | Real-Time Phase | Evaluation |
|---------|---------------|-----------------|------------|
| **Data Processing** | ✅ SMOTE Balancing | ❌ Live Processing | ✅ Metrics Analysis |
| **Model Architecture** | ✅ LSTM-CNN-Attention | ✅ Inference Only | ✅ Performance Plots |
| **Landmark Detection** | ✅ Batch Processing | ✅ Real-Time Tracking | ✅ Accuracy Assessment |
| **Temporal Analysis** | ✅ 75-Frame Sequences | ✅ Sliding Windows | ✅ Confusion Matrix |

## 🔒 System Requirements
### Hardware
- **CPU**: Multi-core processor (Intel i5+ or AMD Ryzen 5+)
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 8GB+ (16GB recommended for training)
- **Storage**: 10GB+ free space for datasets
- **Webcam**: HD camera for real-time detection
### Software
- **OS**: Windows 10+, macOS 10.14+, or Ubuntu 18.04+
- **Python**: 3.8-3.9 (TensorFlow compatibility)
- **CUDA**: 11.2+ (for GPU acceleration)
- **Memory**: Sufficient RAM for model loading (~2GB)
## 🤝 Contributing
### Development Workflow
1. Fork the repository
2. Create feature branch: `git checkout -b feature/seizure-enhancement`
3. Implement improvements to detection accuracy
4. Add comprehensive tests for new features
5. Submit pull request with detailed description
### Research Areas
- **Multi-Modal Fusion**: Combine EEG data with video analysis
- **Edge Deployment**: Optimize for mobile/embedded devices  
- **Dataset Expansion**: Include more diverse seizure types
- **Real-Time Optimization**: Reduce latency for critical applications
## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
## 🙏 Acknowledgments
- **MediaPipe Team**: For robust landmark detection framework
- **TensorFlow Community**: For deep learning infrastructure
- **Medical Research Community**: For epilepsy detection insights
- **Contributors**: M Gopinath, Rahul Shukla
## 📞 Contact
- **Lead Developer**: M Gopinath - mgopinath032398@gmail.com
- **Issues**: GitHub Issues
- **Research Collaboration**: Open to academic partnerships
---
**🔬 Advancing Healthcare through AI** | Documentation | Research Paper
