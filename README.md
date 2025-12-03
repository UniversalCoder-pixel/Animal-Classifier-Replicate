# Animal-Classifier-Replicate
📌 Animal Species Classifier (Deep Learning Demo)

A simple deep-learning demo that classifies animals using a custom-trained ResNet-18 model.
Built with PyTorch and Streamlit.

✨ Features

Classifies 7 animals:
Cat, Cow, Deer, Dog, Elephant, Rabbit, Sheep

Simple UI for uploading images

Custom-trained ResNet18 model

Easy to run locally

🚀 Quick Start
1. Install requirements
pip install -r requirements.txt

2. Train the model (optional)
python main.py


This generates animal_model.pth.

3. Run the demo (UI)
streamlit run app.py

📁 Project Structure
animal_detector_demo/
│── app.py           # Streamlit UI
│── predict.py       # Prediction logic
│── main.py          # Training script
│── animal_model.pth # Saved model
│── requirements.txt
└── dataset/
    ├── train/
    └── test/
This project was developed with assistance from AI tools (ChatGPT) for learning purposes.

