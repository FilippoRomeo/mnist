# MNIST Digit Classification App

An end-to-end handwritten-digit classification project built around PyTorch, Streamlit, PostgreSQL, and Docker. The app lets users draw a digit, inspect the model prediction and confidence, submit corrective feedback, and use stored feedback for later fine-tuning.

## What it includes

- PyTorch MNIST classifier
- Streamlit drawing interface
- Prediction confidence display
- PostgreSQL logging for predictions and feedback
- Feedback-driven incremental retraining
- Docker and Docker Compose deployment
- Conda-based local development workflow

## Architecture

```text
Streamlit UI
    ↓
PyTorch model
    ↓
Prediction + confidence
    ↓
PostgreSQL
    ↓
User feedback / later fine-tuning
```

## Quick start with Docker

Clone the repository:

```bash
git clone https://github.com/FilippoRomeo/mnist.git
cd mnist
```

Create a local `.env` file:

```ini
DB_HOST=localhost
DB_NAME=mnist_db
DB_USER=youruser
DB_PASSWORD=yourpassword
```

Then build and start the containers:

```bash
docker compose up --build
```

Open the app at:

```text
http://localhost:8501
```

## Run locally with Conda

```bash
conda create --name mnist-env python=3.12 -y
conda activate mnist-env
pip install -r requirements.txt
```

Create the PostgreSQL database and initialise the schema:

```bash
psql -U youruser -d postgres -c "CREATE DATABASE mnist_db;"
psql -U youruser -d mnist_db -f init.sql
```

Start Streamlit:

```bash
streamlit run app.py
```

## Database

Predictions and feedback are stored in PostgreSQL. The schema tracks the predicted digit, corrected label, image data, timestamps, and whether feedback has been processed for retraining.

```sql
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    predicted_digit INTEGER NOT NULL,
    true_label INTEGER NOT NULL,
    image BYTEA NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    feedback_processed BOOLEAN DEFAULT FALSE
);
```

## Training

The model can be trained manually with:

```bash
python train.py
```

The application can also use accumulated user feedback for incremental fine-tuning.

## Main stack

`Python` `PyTorch` `Streamlit` `PostgreSQL` `Docker` `Docker Compose`

## Purpose

This project explores the full application loop around a small machine-learning model: training, interactive inference, persistence, user correction, and model improvement rather than classification in isolation.
