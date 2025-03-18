import psycopg2
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from torchvision import transforms
from PIL import Image
import io
from train import MNISTModel,train_on_mnist, save_model, device, MODEL_PATH, retrain_on_single_feedback
# Database connection function
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

def connect_db():
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "mnist_db"),
            user=os.getenv("DB_USER", "filipporomeo"),
            password=os.getenv("DB_PASSWORD", "Macbookpro0-")
        )
        return conn
    except Exception as e:
        st.error(f"❌ Database connection error: {e}")
        return None

# Load model
if "model" not in st.session_state:
    st.session_state.model = MNISTModel().to(device)
    if not os.path.exists(MODEL_PATH):
        st.markdown("Model not found. Training the model...")
        print("Training model on MNIST dataset...")
        train_on_mnist(st.session_state.model)  # Train the model
        save_model(st.session_state.model)  # Save the model to disk
        print(f"Model saved to {MODEL_PATH}")
    else:
        print("Loading existing model...")
        st.session_state.model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    st.session_state.model.eval()  # Set the model to evaluation mode

# Initialize the model
model = MNISTModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
         
# Streamlit UI
st.title('Digit Recognizer')
st.markdown("Draw a digit and let the model predict it!")

# Create two columns for the canvases
col1, col2 = st.columns(2)

# Canvas for drawing
with col1:
    st.write("### Drawing Canvas")
    SIZE = 192
    canvas_result = st_canvas(
        fill_color='#000000',
        stroke_width=15,
        stroke_color='#FFFFFF',
        background_color='#000000',
        width=SIZE,
        height=SIZE,
        drawing_mode="freedraw",
        key='canvas'
    )

# Canvas for displaying the processed image
with col2:
    st.write("### Processed Image")
    if canvas_result.image_data is not None:
        img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_pil = Image.fromarray(img_gray)
        st.image(img_pil, caption="Processed Image (28x28)", width=100)

# Prediction logic
if canvas_result.image_data is not None and st.button("🔍 Predict"):
    try:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        img_tensor = transform(img_pil).unsqueeze(0).to(device, dtype=torch.float32)

        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
            prediction = int(np.argmax(probabilities))

        st.write(f"### 🎯 Model Prediction: {prediction}")
        st.bar_chart(probabilities)

        # Store in session state
        st.session_state.prediction = prediction
        st.session_state.probabilities = probabilities
        st.session_state.img_data = img_pil

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")

# Feedback Submission
if "prediction" in st.session_state:  # Fixed typo here
    true_label = st.number_input("✔️ Enter the correct digit (if wrong):", min_value=0, max_value=9, step=1)

    if st.button("📤 Submit Feedback"):
        try:
            img_bytes = io.BytesIO()
            st.session_state.img_data.save(img_bytes, format='PNG')
            img_data = img_bytes.getvalue()

            conn = connect_db()
            if conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO predictions (predicted_digit, true_label, image, feedback_processed) VALUES (%s, %s, %s, %s)",
                    (st.session_state.prediction, true_label, psycopg2.Binary(img_data), False)
                )
                conn.commit()
                cursor.close()
                conn.close()
                st.success("✅ Feedback stored successfully!")
                st.rerun()  # Auto-refresh the UI
        except Exception as e:
            st.error(f"❌ Error during feedback submission: {e}")

# Fetch and Display Saved Predictions
st.subheader("Recent Predictions")
conn = connect_db()
if conn:
    cursor = conn.cursor()
    cursor.execute("SELECT id, predicted_digit, true_label, image, timestamp, created_at FROM predictions ORDER BY created_at DESC LIMIT 5")
    rows = cursor.fetchall()

    if rows:
        for row in rows:
            id, predicted, true_label, img_data, timestamp, created_at = row
            img = Image.open(io.BytesIO(img_data))

            # Display the image with prediction details
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(img, caption=f"ID {id}", width=100)
            with col2:
                st.write(f"**Predicted:** {predicted}  |  **True:** {true_label}")
                st.write(f"**Timestamp:** {timestamp}")
                st.write(f"**Created at:** {created_at}")

                
            # Edit true label
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                new_label = st.number_input(f"Edit true label for ID {id}:", min_value=0, max_value=9, value=true_label, key=f"edit_{id}")
            with col2:
                if st.button(f"💾 Update ID {id}", key=f"update_{id}"):
                    try:
                        update_conn = connect_db()
                        if update_conn:
                            update_cursor = update_conn.cursor()
                            update_cursor.execute("UPDATE predictions SET true_label = %s WHERE id = %s", (new_label, id))
                            update_conn.commit()
                            update_cursor.close()
                            update_conn.close()
                            st.success(f"✅ Updated ID {id} to {new_label}")
                            st.rerun()  # Refresh the UI automatically
                    except Exception as e:
                        st.error(f"❌ Update failed: {e}")
            with col3:
                # Delete option
                if st.button(f"🗑️ Delete ID {id}", key=f"delete_{id}"):
                    try:
                        delete_conn = connect_db()
                        if delete_conn:
                            delete_cursor = delete_conn.cursor()
                            delete_cursor.execute("DELETE FROM predictions WHERE id = %s", (id,))
                            delete_conn.commit()
                            delete_cursor.close()
                            delete_conn.close()
                            st.warning(f"🗑️ Deleted entry ID {id}")
                            st.rerun()  # Refresh the UI automatically
                    except Exception as e:
                        st.error(f"❌ Deletion failed: {e}")

            # Retrain button
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button(f"🔄 Retrain Model for ID {id}", key=f"retrain_{id}"):
                    try:
                        # Retrain the model until prediction matches true value
                        while True:
                            # Retrain the model on the specific feedback data
                            model = retrain_on_single_feedback(model, img, true_label)
                            
                            # Perform prediction on the current image
                            img_gray = img.convert('L')
                            img_resized = img_gray.resize((28, 28))
                            transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ])
                            img_tensor = transform(img_resized).unsqueeze(0).to(device, dtype=torch.float32)

                            with torch.no_grad():
                                output = model(img_tensor)
                                probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
                                new_prediction = int(np.argmax(probabilities))

                            # Display new prediction
                            st.write(f"### 🎯 New Prediction for ID {id}: {new_prediction}")
                            st.bar_chart(probabilities)

                            # Check if prediction matches true value
                            if new_prediction == true_label:
                                st.success(f"✅ Model retrained successfully! Prediction now matches true value: {true_label}")
                                break
                            else:
                                st.write(f"Retraining... Current prediction: {new_prediction}")

                    except Exception as e:
                        st.error(f"❌ Error during retraining: {e}")
            with col2:
                if st.button(f"🔍 Redo Prediction for ID {id}", key=f"redo_{id}"):
                    try:
                        # Preprocess the image
                        img_gray = img.convert('L')
                        img_resized = img_gray.resize((28, 28))
                        transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])
                        img_tensor = transform(img_resized).unsqueeze(0).to(device, dtype=torch.float32)

                        # Perform prediction
                        with torch.no_grad():
                            output = model(img_tensor)
                            probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
                            new_prediction = int(np.argmax(probabilities))

                        # Display new prediction
                        st.write(f"### 🎯 New Prediction for ID {id}: {new_prediction}")
                        st.bar_chart(probabilities)

                        # Check if prediction changed
                        if new_prediction != predicted:
                            update_conn = connect_db()
                            if update_conn:
                                update_cursor = update_conn.cursor()
                                update_cursor.execute(
                                    "UPDATE predictions SET predicted_digit = %s WHERE id = %s",
                                    (new_prediction, id)
                                )
                                update_conn.commit()
                                update_cursor.close()
                                update_conn.close()
                                st.success(f"✅ Updated prediction for ID {id} to {new_prediction}")
                                st.rerun()  # Refresh UI automatically

                    except Exception as e:
                        st.error(f"❌ Error during prediction: {e}")

    else:
        st.write("No saved corrections yet.")
    cursor.close()
    conn.close()