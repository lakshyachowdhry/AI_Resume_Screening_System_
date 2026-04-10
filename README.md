## AI Resume Screening System

This project is an end-to-end AI-based resume screening system built as a final-year project. It combines PDF parsing, NLP preprocessing, similarity scoring, skill analysis, and a simple machine learning model, all integrated into an interactive Streamlit application.

---

## Features

- Upload multiple PDF resumes at once  
- Extract text from resumes using PyPDF2  
- NLP preprocessing using NLTK (tokenization, stopword removal, cleaning)  
- Skill extraction and gap analysis based on job description  
- Resume ranking using TF-IDF cosine similarity  
- Optional machine learning model (Logistic Regression) for prediction  
- Combined scoring system (similarity + skills + model probability)  
- Duplicate resume detection  
- Candidate-wise explanation for transparency  
- Download results as CSV  
- Logging of key events  

---

## Project Structure


app.py # Streamlit application
parser.py # PDF text extraction
preprocess.py # Text preprocessing
skills.py # Skill extraction and gap analysis
similarity.py # TF-IDF and similarity logic
model.py # Model training script
evaluator.py # Model evaluation
utils.py # Helper functions

data/
dataset.csv # Training dataset

model/
model.pkl # Trained model
vectorizer.pkl # TF-IDF vectorizer

outputs/
logs.txt # Logs


---

## Setup

### 1. Create a virtual environment (recommended)

```bash
python -m venv .venv
.venv\Scripts\activate
2. Install dependencies
pip install -r requirements.txt
Training the Model

The model is a Logistic Regression classifier trained on resume–job description pairs.

Dataset format (data/dataset.csv):

Resume_Text
Job_Description
Label (1 = good match, 0 = poor match)

Run the training:

python model.py

This will generate:

model/model.pkl
model/vectorizer.pkl
Running the App
streamlit run app.py

Steps:

Enter or select a job description
Upload one or more resumes (PDF)
Click Analyze Candidates
Output
Ranked list of candidates
Top candidate highlighted
Score breakdown (TF-IDF, skills, model)
Skill gap analysis
Resume summaries
Duplicate detection (if any)
CSV download option
Notes
Make sure the model/ folder contains model.pkl and vectorizer.pkl
If not, run python model.py before starting the app
NLTK resources will download automatically on first run
Demo Tips
Use 3–5 sample resumes
Try different job roles (Data Scientist, AI/ML Engineer, etc.)
Show evaluation metrics from sidebar
Explain pipeline: parsing → preprocessing → similarity → scoring
Deployment

The app can be deployed on platforms like Streamlit Cloud.

Before deploying:

Ensure model files are present
Add requirements.txt
(Optional) Add API key in secrets for GenAI features
Author

Lakshya Chowdhry