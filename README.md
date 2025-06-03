# COVID-19 X-ray Detection

An AI-powered application that uses deep learning to detect COVID-19 from chest X-ray images.

## About This Project

This project uses a Convolutional Neural Network (CNN) to classify chest X-ray images as either showing signs of COVID-19 or not. The model was trained on a dataset of X-ray images and achieves good accuracy in distinguishing between COVID-19 positive and negative cases.


## Features

- Upload chest X-ray images for analysis
- Real-time prediction using a trained CNN model
- User-friendly interface with clear results
- Detailed prediction confidence visualization

## Try It Online

You can try the application directly in your browser:
[COVID-19 X-ray Detection App](https://covid-19-xray-detection.streamlit.app/)

## How It Works

1. The app takes a chest X-ray image as input
2. The image is preprocessed to match the format expected by the model (grayscale, 224x224 pixels)
3. The trained CNN model analyzes the image
4. Results are displayed showing the probability of COVID-19 presence

## Model Architecture

The model uses a CNN architecture with:
- Multiple convolutional layers with ReLU activation
- Max pooling layers
- Dropout layers to prevent overfitting
- Dense layers for classification
- Binary output (COVID-19 positive or negative)

## Local Setup

### Prerequisites
- tensorflow: 2.18.0
- numpy: 2.0.2
- opencv-python-headless: 4.11.0.86
- cv2: 4.11.0
- pillow: 11.2.1
- PIL: 11.2.1

### Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/COVID-19-Xray-Detection.git
cd COVID-19-Xray-Detection
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run covid19_xray_app.py
```

## Dataset

The model was trained on the [COVID-19 X-ray Dataset](https://www.kaggle.com/datasets/khoongweihao/covid19-xray-dataset-train-test-sets) from Kaggle, which contains both COVID-19 positive and negative chest X-ray images.

## Disclaimer

This application is for educational and research purposes only. It is not intended for medical diagnosis. The predictions made by this model should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

## License

MIT License
