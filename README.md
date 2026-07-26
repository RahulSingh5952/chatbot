# Study Helper Chatbot Web App

A simple browser chatbot built with Python, Flask, and classic machine learning.
The project keeps the frontend basic and focuses on the AI/ML core.

## What it uses

- Python
- Flask
- pandas
- scikit-learn
- a CSV dataset of `Question,Answer` pairs

## Project structure

- `app.py` - loads the dataset, trains the model, and serves the web app
- `chatbot_dataset.csv` - training data for the chatbot
- `templates/index.html` - main page
- `static/style.css` - page styling
- `static/app.js` - browser chat behavior

## How it works

1. The app reads `chatbot_dataset.csv`.
2. Questions are normalized and used to train a text classifier.
3. The web page sends messages to `/chat`.
4. The model returns the best matching answer.
5. If confidence is low, the bot gives a fallback response.

## Run it

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
python app.py
```

Open the browser at:

```text
http://127.0.0.1:5000
```

## Dataset format

The CSV must contain exactly these two columns:

```csv
Question,Answer
hi,Hello!
what is ai,AI means Artificial Intelligence.
```

## Notes

- The fullstack part is intentionally simple.
- The main focus is the chatbot logic and ML pipeline.
- You can expand the dataset later without changing the app code.
