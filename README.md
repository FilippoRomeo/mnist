Overview

This project is an end-to-end MNIST digit classification application designed for deployment on a self-managed server. It involves training a PyTorch model, creating an interactive front-end with Streamlit, logging predictions to a PostgreSQL database, and deploying the application using Docker and Docker Compose.

Features

Digit Recognition: A PyTorch model classifies handwritten digits from the MNIST dataset.

Interactive Web UI: Users can draw digits in a Streamlit-based web app.

Prediction & Feedback: The app provides model predictions along with confidence scores, and users can submit correct labels for feedback.

Database Logging: PostgreSQL stores predictions and feedback.

Retraining: The model can be fine-tuned incrementally based on user feedback.

Containerization & Deployment: The entire application is containerized using Docker and deployed on a self-managed server.

Technologies Used

Machine Learning: PyTorch

Front-End: Streamlit

Database: PostgreSQL

Containerization: Docker, Docker Compose

Installation & Setup

Prerequisites

Ensure you have the following installed:

Python 3.x

Docker & Docker Compose

PostgreSQL

Clone the Repository

git clone https://github.com/FilippoRomeo/mnist.git
cd mnist-digit-recognizer

Set Up Environment Variables

Create a .env file with the following variables:

DB_HOST=localhost
DB_NAME=mnist_db
DB_USER=youruser
DB_PASSWORD=yourpassword

Train the Model

Run the training script:

python train.py

Run the Application with Docker

Build and start the containers:

docker-compose up --build

Access the Application

Once the containers are running, open your browser and go to:

http://localhost:8501

Database Schema

The PostgreSQL database stores predictions and feedback:

CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    predicted_digit INT,
    true_label INT,
    image BYTEA,
    feedback_processed BOOLEAN DEFAULT FALSE,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

Deployment

To deploy the application on a self-managed server:

Set up the server and install Docker.

Clone the repository onto the server.

Configure environment variables.

Run docker-compose up --build to start the app.

Ensure the app is accessible via a public IP or domain.