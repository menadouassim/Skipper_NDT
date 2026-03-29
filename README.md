# Skipper NDT - Non-Destructive Testing Pipe Detection

A machine learning project for detecting and analyzing pipes in Non-Destructive Testing (NDT) imagery. This repository contains data preparation pipelines, model training, and analysis notebooks for pipe detection using computer vision techniques.

## 📋 Project Overview

This project focuses on building and training machine learning models to detect pipes in Skipper_NDT (Non-Destructive Testing) provided images. The pipeline includes data preparation, model training with TensorFlow, and analysis of detection results including centerline calculations and width measurements.

## 🗂️ Repository Structure

```
├── task_number01.ipynb           # CNN for binary classification
├── task_number02.ipynb           # pipeline for scans width calculation
├── task_number03.ipynb           # CNN with augmentation for intensity check
├── task_number05.ipynb           # pipeline to check if the scan is off-center
├── data_preparation.ipynb        # Data preparation and augmentation pipeline
├── requirements.txt              # Python dependencies
├── env/                          # Virtual environment
└── README.md                     # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip (Python package manager)
- GPU support recommended (code includes GPU configuration)

### Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Skipper_NDT
   ```

2. **Create a virtual environment (if not already created):**
   ```bash
   python3 -m venv env
   ```

3. **Activate the virtual environment:**
   ```bash
   # On macOS/Linux:
   source env/bin/activate
   
   # On Windows:
   env\Scripts\activate
   ```

4. **Upgrade pip:**
   ```bash
   pip install --upgrade pip
   ```

5. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

6. **Verify installation:**
   ```bash
   python -c "import tensorflow; import numpy; import pandas; print('All dependencies installed successfully!')"
   ```

## 📦 Dependencies

- **pandas** - Data manipulation and CSV processing
- **numpy** - Numerical computations
- **scikit-learn** - Machine learning utilities and train/test splitting
- **matplotlib** - Data visualization
- **scikit-image** - Image processing (resizing, transformations)
- **tensorflow** - Deep learning framework for model training

## 📓 Notebooks

### Data Preparation
- **`data_preparation.ipynb`** - Prepares and augments the NDT image dataset
- **`task_number01.ipynb`** - Initial data loading, extraction, and setup

### Model Training
- **`task_number01.ipynb`** - Training phase 1
- **`task_number03.ipynb`** - Training phase 3


### Analysis & Visualization
- **`task_number02.ipynb`** - Experiments with pipe width calculations
- **`task_number05.png`** - Output visualization showing detected pipe centerlines and scans positioning

## 🎯 Key Features

- **GPU Support**: Automatically configures and uses available GPU (with multi-GPU support)
- **Data Splitting**: Implements train/test split with configurable ratios
- **Image Processing**: Resizes and processes .Npz images for model input
- **Centerline Detection**: Analyzes and visualizes pipe centerlines
- **Width Calculation**: Measures pipe widths from detected regions

## 💻 Usage

1. **Run data preparation:**
   ```bash
   jupyter notebook data_preparation.ipynb
   ```

2. **Execute training tasks :**
   ```bash
   jupyter notebook task_number01.ipynb
   jupyter notebook task_number02.ipynb
   jupyter notebook task_number03.ipynb
   jupyter notebook task_number05.ipynb
   ```

3. **Analyze results:**
   each task has the results plotted and printed underneath

## 🔧 Configuration

Key configuration parameters found in the notebooks:
- **SEED**: Random seed for reproducibility (default: 42)
- **SPLIT**: Train/test split ratio (default: 0.2)
- **Data paths**: Configured for both local and cloud storage (Google Drive support)

## 📊 Data Format

The project expects:
- NPZ images in ZIP archives
- Labels in CSV format (`pipe_detection_label.csv`) with fields:
  - `field_file`: Image filename
  - `label`: Classification label

## 🖥️ GPU Configuration

The notebooks include GPU configuration code:
```python
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[1], 'GPU')  # Uses second GPU
```

Modify the GPU index as needed for your system.

## 📝 Notes

- Ensure sufficient disk space for extracted training data
- GPU memory requirements depend on image resolution and batch size
- Virtual environment is preconfigured with all necessary packages

## 👤 Author

Created by: Ahmed Ouassim Menad 

