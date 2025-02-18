# 🌾 Wheat Disease Detection with Machine Learning

This project uses **deep learning and computer vision** to detect **wheat rust disease** from leaf images, helping farmers and agricultural experts diagnose crop health efficiently.

##  Features
- **Image Classification**: Detects **Healthy, Mild Rust, and Severe Rust** wheat leaves.
- **Real-Time Prediction**: Users can upload images for instant analysis.
- **Flask Web App**: A user-friendly interface for disease detection.
- **High Accuracy**: Achieves **92.5% classification accuracy** using TensorFlow.

## Technologies Used
- **Machine Learning**: TensorFlow, OpenCV, Scikit-Learn
- **Web Application**: Flask
- **Data Handling**: NumPy, Pandas

##  Project Structure

wheat-disease-detection/ │── dataset/ # Training dataset (Healthy, Mild Rust, Severe Rust) │── src/ │ ├── data_loader.py # Loads and preprocesses images │ ├── model.py # Defines the CNN model │ ├── train.py # Trains the model │ ├── evaluate.py # Evaluates model accuracy │── app.py # Flask backend for real-time prediction │── requirements.txt # Project dependencies │── README.md # Project documentation │── templates/ # HTML UI for web interface │── static/uploads/ # Stores uploaded images


##  Installation & Setup
1. **Clone the repository**  
   ```bash
   git clone https://github.com/kidusaman/wheat-disease-detection.git
   cd wheat-disease-detection
Install dependencies
pip install -r requirements.txt

## Install dependencies
pip install -r requirements.txt

## Run the Flask web app
python app.py

## Access the app
Open your browser and go to:
http://127.0.0.1:5000/
