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

### Train the Model

Run the training script:

```bash
python train.py
```

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

## Deployment

To deploy the application on a self-managed server:

1. Set up the server and install Docker.
2. Clone the repository onto the server.
3. Configure environment variables.
4. Run `docker-compose up --build` to start the app.
5. Ensure the app is accessible via a public IP or domain.

## Docker Configuration

**Docker Compose File:**

```yaml
version: '3.8'

services:
  db:
    image: postgres:15
    container_name: mnist-db
    environment:
      POSTGRES_DB: ${DB_NAME}
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - mnist-net
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER} -d ${DB_NAME}"]
      interval: 5s
      timeout: 5s
      retries: 5

  app:
    build: .
    container_name: mnist-app
    ports:
      - "8501:8501"
    environment:
      - DB_HOST=db
      - DB_NAME=${DB_NAME}
      - DB_USER=${DB_USER}
      - DB_PASSWORD=${DB_PASSWORD}
    depends_on:
      db:
        condition: service_healthy
    networks:
      - mnist-net

volumes:
  postgres_data:

networks:
  mnist-net:
    driver: bridge
```

**Dockerfile:**

```dockerfile
# Use Python 3.12 as the base image
FROM python:3.12-slim

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port that Streamlit runs on
EXPOSE 8501

# Set the environment variable for Streamlit
ENV STREAMLIT_SERVER_PORT=8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Requirements File

Below is the list of dependencies used in this project:

```ini
absl-py==2.1.0
altair==5.5.0
astunparse==1.6.3
attrs==25.1.0
blinker==1.9.0
cachetools==5.5.2
certifi==2025.1.31
charset-normalizer==3.4.1
click==8.1.8
contourpy==1.3.1
cycler==0.12.1
filelock==3.17.0
flatbuffers==25.2.10
fonttools==4.56.0
fsspec==2025.3.0
gast==0.6.0
gitdb==4.0.12
GitPython==3.1.44
google-pasta==0.2.0
grpcio==1.70.0
h5py==3.13.0
idna==3.10
Jinja2==3.1.6
jsonschema==4.23.0
keras==3.9.0
matplotlib==3.10.1
numpy==2.0.2
opencv-python==4.11.0.86
pandas==2.2.3
pillow==11.1.0
psycopg2-binary==2.9.10
streamlit==1.43.1
tensorflow==2.18.0
torch==2.6.0
torchaudio==2.6.0
torchvision==0.21.0
```

This guide provides all necessary steps to set up and deploy the MNIST classification project efficiently. 🚀

