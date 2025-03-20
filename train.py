import psycopg2
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import io

# Database connection function
def connect_db(retries=5, delay=3):
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(
                host=os.getenv("DB_HOST", "localhost"),
                database=os.getenv("DB_NAME"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                port="5432"
            )
            return conn
        except psycopg2.OperationalError as e:
            if "could not translate host name" in str(e):
                st.error(f"❌ DNS resolution failed (attempt {attempt+1}/{retries})")
            else:
                st.error(f"❌ Connection failed (attempt {attempt+1}/{retries}): {e}")
            time.sleep(delay)
    return None
# Define the model
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 300)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(300, 50)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x  # Raw logits

# Set device (MPS or CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load model
MODEL_PATH = 'mnist_model.pth'

# Fetch feedback data
def fetch_feedback_data():
    conn = connect_db()
    if conn:
        cursor = conn.cursor()
        cursor.execute("SELECT image, true_label FROM predictions WHERE feedback_processed = False")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows
    return []

# Preprocess images and labels
def preprocess_images(rows):
    images = []
    labels = []
    for row in rows:
        img_data = row[0]
        true_label = row[1]
        
        img = Image.open(io.BytesIO(img_data))
        img_gray = img.convert('L')
        img_resized = img_gray.resize((28, 28))
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        img_tensor = transform(img_resized)
        
        images.append(img_tensor)
        labels.append(true_label)
    
    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return images, labels

# Train the model on feedback data
def train_on_feedback(model, images, labels, epochs=1, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Move images and labels to the same device as the model
    images = images.to(device)
    labels = labels.to(device)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    return model

# Save the updated model
def save_model(model, path='mnist_model.pth'):
    torch.save(model.state_dict(), path)

# Main function to retrain the model
def train_model_on_feedback(model):
    # Fetch feedback data
    feedback_data = fetch_feedback_data()
    
    if not feedback_data:
        print("No new feedback data to train on.")
        return
    
    # Preprocess images and labels
    images, labels = preprocess_images(feedback_data)
    
    # Train the model
    model = train_on_feedback(model, images, labels)
    
    # Save the updated model
    save_model(model)
    
    # Mark feedback as processed
    conn = connect_db()
    if conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE predictions SET feedback_processed = True WHERE feedback_processed = False")
        conn.commit()
        cursor.close()
        conn.close()
    
    print("Model trained on feedback data and updated successfully.")

# Train the model on MNIST dataset
def train_on_mnist(model, epochs=5, batch_size=64, lr=0.001):
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Train the model
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")
    
    print("Training on MNIST dataset completed.")

def retrain_on_single_feedback(model, image, true_label, epochs=1, lr=0.001):
    """
    Retrain the model on a single feedback image and true label.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Preprocess the image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    img_tensor = transform(image).unsqueeze(0).to(device, dtype=torch.float32)
    label_tensor = torch.tensor([true_label], dtype=torch.long).to(device)
    
    # Train the model
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(img_tensor)
        loss = criterion(outputs, label_tensor)
        loss.backward()
        optimizer.step()
    
    # Save the updated model
    save_model(model)
    
    return model

# Run the training
if __name__ == "__main__":
    # Initialize the model
    model = MNISTModel().to(device)
    
    # Check if the model already exists
    if os.path.exists(MODEL_PATH):
        print("Loading existing model...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print("Training model on MNIST dataset...")
        train_on_mnist(model)
        save_model(model)
        print(f"Model saved to {MODEL_PATH}")
    
    model.eval()
    
    # Train on feedback data
    train_model_on_feedback(model)