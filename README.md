# Overview

This project is an end-to-end MNIST digit classification application designed for deployment on a self-managed server. It involves training a PyTorch model, creating an interactive front-end with Streamlit, logging predictions to a PostgreSQL database, and deploying the application using Docker and Docker Compose. I also added the possibility to re-train on the wrong answers to have a better model afterward. I'm using MPS because I am on an Apple device, so adjust the code to your needs.

## Features

- **Digit Recognition**: A PyTorch model classifies handwritten digits from the MNIST dataset.
- **Interactive Web UI**: Users can draw digits in a Streamlit-based web app.
- **Prediction & Feedback**: The app provides model predictions along with confidence scores, and users can submit correct labels for feedback.
- **Database Logging**: PostgreSQL stores predictions and feedback.
- **Retraining**: The model can be fine-tuned incrementally based on user feedback.
- **Containerization & Deployment**: The entire application is containerized using Docker and deployed on a self-managed server.

## Technologies Used

- **Machine Learning**: PyTorch
- **Front-End**: Streamlit
- **Database**: PostgreSQL
- **Containerization**: Docker, Docker Compose

## Installation & Setup

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- Docker & Docker Compose
- PostgreSQL

### Clone the Repository

```bash
git clone https://github.com/FilippoRomeo/mnist.git
cd mnist
