# Overview

This project is an end-to-end MNIST digit classification application designed for deployment on a self-managed server. It involves training a PyTorch model, creating an interactive front-end with Streamlit, logging predictions to a PostgreSQL database, and deploying the application using Docker and Docker Compose. Additionally, the model can be fine-tuned based on user feedback to improve accuracy. The development environment uses Conda for managing dependencies.

## Features

- **Digit Recognition**: A PyTorch model classifies handwritten digits from the MNIST dataset.
- **Interactive Web UI**: Users can draw digits in a Streamlit-based web app.
- **Prediction & Feedback**: The app provides model predictions along with confidence scores, and users can submit correct labels for feedback.
- **Database Logging**: PostgreSQL stores predictions and feedback.
- **Retraining**: The model can be fine-tuned incrementally based on user feedback.
- **Containerization & Deployment**: The entire application is containerized using Docker and deployed on a self-managed server.

## Technologies Used

- **Machine Learning**: PyTorch, TensorFlow
- **Front-End**: Streamlit
- **Database**: PostgreSQL
- **Containerization**: Docker, Docker Compose
- **Virtual Environment**: Conda

## Installation & Setup

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- Conda (Miniconda or Anaconda)
- Docker & Docker Compose
- PostgreSQL

### Clone the Repository

```bash
git clone https://github.com/FilippoRomeo/mnist.git
cd mnist
```

### Set Up the Conda Environment

Create and activate the Conda virtual environment:

```bash
conda create --name mnist-env python=3.12
conda activate mnist-env
```

### Install Dependencies

Install all required dependencies:

```bash
pip install -r requirements.txt
```

### Set Up Environment Variables

Create a `.env` file with the following variables:

```ini
DB_HOST=localhost
DB_NAME=mnist_db
DB_USER=youruser
DB_PASSWORD=yourpassword
```

### Set Up the PostgreSQL Database

If you are running PostgreSQL locally, create the database:

```bash
psql -U youruser -d postgres -c "CREATE DATABASE mnist_db;"
```

Run the database schema initialization script:

```bash
psql -U youruser -d mnist_db -f init.sql
```

Alternatively, if using Docker, the database will be set up automatically with the following command.

### Run the Application with Docker

Build and start the containers:

```bash
docker-compose up --build
```

### Access the Application

Once the containers are running, open your browser and go to:

[http://localhost:8501](http://localhost:8501)

## Database Schema

The PostgreSQL database stores predictions and feedback:

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

### Train the Model

You don't need to run the training script because it is automatically executed by app.py, but to eventually test it run:

```bash
python train.py
```

## Deployment

To deploy the application on a self-managed server:

1. Set up the server and install Docker.
2. Clone the repository onto the server.
3. Configure environment variables.
4. Run `docker-compose up --build` to start the app.
5. Ensure the app is accessible via a public IP or domain.


This guide provides all necessary steps to set up and deploy the MNIST classification project efficiently. 🚀

