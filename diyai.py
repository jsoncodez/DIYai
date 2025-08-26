"""
DIYai - Do It Yourself - AI Model Builder
---------------------
A GUI-based image classification tool using TensorFlow and Keras.

Features:
- Create and manage custom image categories (datasets)
- Upload and auto-rename images into categorized folders
- Train a convolutional neural network (CNN) or pretrained model
- Predict the category of a new image with class probabilities
- Add misclassified images to correct categories for continuous improvement
- Optionally fine-tune the model by adding dataset
- GUI built with Tkinter

Recommended for:
- Beginners learning computer vision and machine learning
- Small-scale custom image classification tasks

Author: Jason S. -jscodez
Date: [2025-08-25]
Python Version: 3.x
TensorFlow Version: 2.x
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tkinter import Tk, filedialog, simpledialog, Button, Label, messagebox, Toplevel, StringVar, OptionMenu
from tkinter.ttk import Combobox

# Global variables
IMAGE_SIZE = 224
MODEL = None
DATASET_DIR = "dataset"


# Create dataset root if not exists
os.makedirs(DATASET_DIR, exist_ok=True)


# Function to Create Class Folder
def create_class_folder(class_name):
    class_path = os.path.join(DATASET_DIR, class_name)
    if not os.path.exists(class_path):
        os.makedirs(class_path)
        messagebox.showinfo("Class Created", f"Class '{class_name}' created.")
    else:
        messagebox.showwarning("Class Exists", f"Class '{class_name}' already exists.")


# Function to upload images and rename for organization
def upload_images(class_name):
    class_path = os.path.join(DATASET_DIR, class_name)
    if not os.path.exists(class_path):
        messagebox.showwarning("Class Missing", f"Class '{class_name}' doesn't exist.")
        return

    file_paths = filedialog.askopenfilenames(
        title="Select Images",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.gif")]
    )

    if not file_paths:
        messagebox.showwarning("No File", "No files selected.")
        return

    existing_files = os.listdir(class_path)
    count = len(existing_files)

    for idx, file_path in enumerate(file_paths, start=1):
        img = cv2.imread(file_path)
        if img is None:
            continue

        img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        new_filename = f"{class_name}_{count + idx:03d}.jpg"
        save_path = os.path.join(class_path, new_filename)
        cv2.imwrite(save_path, img_resized)

    messagebox.showinfo("Upload Complete", f"{len(file_paths)} image(s) uploaded to '{class_name}'.")


# Function to load dataset from dataset folder or create new dataset class
def load_dataset():
    images = []
    labels = []
    class_names = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])

    for idx, class_name in enumerate(class_names):
        class_folder = os.path.join(DATASET_DIR, class_name)
        for img_name in os.listdir(class_folder):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                img_path = os.path.join(class_folder, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                    images.append(img_resized)
                    labels.append(idx)

    if not images:
        raise ValueError("No valid images found. Check dataset folders.")

    return np.array(images) / 255.0, np.array(labels), class_names


# Function to start building model - augments images to standardize and potential accuracy of model
def build_model(class_count):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(class_count, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Step 5: Train Model
def train_model():
    global MODEL
    try:
        images, labels, class_names = load_dataset()
        MODEL = build_model(len(class_names))
        MODEL.fit(images, labels, epochs=10, batch_size=32)
        #MODEL.save("diyai_model.h5")
        MODEL.save('diyai_model.keras')
        messagebox.showinfo("Training Complete", "Model trained and saved.")
    except Exception as e:
        messagebox.showerror("Training Error", str(e))



# Helper Function - augmenting user's uploaded prediction image
def apply_clahe(img):
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    img_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return img_clahe

# Helper Function - augmenting user's uploaded prediction image - resize
def resize_with_padding(img, size=224):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h))

    top = (size - new_h) // 2
    bottom = size - new_h - top
    left = (size - new_w) // 2
    right = size - new_w - left

    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
    return img_padded

# Helper function - Generate augmented images for test-time augmentation (TTA)
def augment_image(img):
    augmented_images = []

    # Original
    augmented_images.append(img)

    # Flip horizontally
    augmented_images.append(cv2.flip(img, 1))

    # Rotate (+/-) 15 degrees
    for angle in [-15, 15]:
        M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1)
        rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        augmented_images.append(rotated)

    return np.array(augmented_images)


# Function to take user input image to predict, basic image augmentation
def predict_image():
    global MODEL
    CONFIDENCE_THRESHOLD = 80  # percent

    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        messagebox.showwarning("No File", "No image selected.")
        return

    img = cv2.imread(file_path)
    if img is None:
        messagebox.showerror("Invalid Image", "Could not read image.")
        return

    img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)) / 255.0
    img_input = np.expand_dims(img_resized, axis=0)

    if MODEL is None:
        try:
            #MODEL = tf.keras.models.load_model('diyai_model.h5')
            MODEL = tf.keras.models.load_model('diyai_model.keras')
        except Exception as e:
            messagebox.showerror("Model Load Error", str(e))
            return

    predictions = MODEL.predict(img_input)[0]
    class_names = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])

    sorted_indices = np.argsort(predictions)[::-1]
    top_class_index = sorted_indices[0]
    top_confidence = predictions[top_class_index] * 100

    # Displays confidence levels of each class
    results = []
    if top_confidence < CONFIDENCE_THRESHOLD:
        results.append(f"Unknown: {100 - top_confidence:.2f}% (most likely: {class_names[top_class_index]})")
    for idx in sorted_indices:
        results.append(f"{class_names[idx]}: {predictions[idx]*100:.2f}%")

    result_text = "\n".join(results)

    # Get user feedback to improve model and accuracy - user has option to correct class or confirm or cancel
    feedback_window = Toplevel()
    feedback_window.title("Prediction Feedback")

    Label(feedback_window, text="Prediction Results:", font=("Arial", 12, "bold")).pack(pady=5)
    Label(feedback_window, text=result_text, justify="left").pack(pady=5)

    Label(feedback_window, text="Is this prediction correct?").pack(pady=10)

    def add_to_class(class_name):
        # Save resized original image to class folder
        class_path = os.path.join(DATASET_DIR, class_name)
        os.makedirs(class_path, exist_ok=True)

        # rename file and add to correct class directory
        existing_files = os.listdir(class_path)
        count = len(existing_files)
        new_filename = f"{class_name}_{count + 1:03d}.jpg"
        save_path = os.path.join(class_path, new_filename)

        # Save original image resized to IMAGE_SIZE (not normalized)
        img_save = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        cv2.imwrite(save_path, img_save)

        messagebox.showinfo("Image Added", f"Image added to class '{class_name}'.  Must Re-Train model to implement changes to dataset")
        feedback_window.destroy()

    # GUI and Function - if user confirms prediction was correct.
    def on_confirm():
        # User confirms prediction is correct
        add_to_class(class_names[top_class_index])

    # GUI and Function - allows user to add new image to dataset and correct its class for improving dataset and accuracy
    def on_correct():
        for widget in feedback_window.winfo_children():
            widget.destroy()

        Label(feedback_window, text="Select the correct class:").pack(pady=5)

        selected_class = StringVar(feedback_window)
        combo = Combobox(feedback_window, textvariable=selected_class, values=class_names)
        combo.pack(pady=5)
        combo.set(class_names[0])

        def on_correct_confirm():
            chosen_class = selected_class.get()
            if chosen_class:
                add_to_class(chosen_class)
            else:
                messagebox.showwarning("Selection Error", "Please select a class.")

        Button(feedback_window, text="Confirm", command=on_correct_confirm).pack(pady=10)
        Button(feedback_window, text="Cancel", command=feedback_window.destroy).pack()

    # Function Button - User does not add image
    def on_cancel():

        feedback_window.destroy()

    # Buttons for feedback
    Button(feedback_window, text="Yes, correct", command=on_confirm).pack(side="left", padx=10, pady=10)
    Button(feedback_window, text="No, select correct class", command=on_correct).pack(side="left", padx=10, pady=10)
    Button(feedback_window, text="Cancel", command=on_cancel).pack(side="left", padx=10, pady=10)


# GUI setup via TK library
def create_ui():
    window = Tk()
    window.title("DIYai")

    label = Label(window, text="DIYai", font=("Arial", 16))
    label.pack(pady=20)

    Button(window, text="Create New Class", command=create_class_interface).pack(pady=10)
    Button(window, text="Upload Images", command=upload_images_interface).pack(pady=10)
    Button(window, text="Train Model", command=train_model).pack(pady=10)
    Button(window, text="Predict Image", command=predict_image).pack(pady=10)

    window.mainloop()


# GUI - to create a new class and select photos from user files
def create_class_interface():
    class_name = simpledialog.askstring("Enter Class Name", "Enter name of the new class:")
    if class_name:
        create_class_folder(class_name.strip())


def upload_images_interface():
    class_folders = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])

    top = Toplevel()
    top.title("Select or Create Class")

    Label(top, text="Select or type a class name:").pack(pady=10)

    selected_class = StringVar(top)
    combo = Combobox(top, textvariable=selected_class, values=class_folders)
    combo.pack(pady=10)
    combo.set(class_folders[0] if class_folders else "")  # default selection

    def on_confirm():
        class_name = selected_class.get().strip()
        if not class_name:
            messagebox.showwarning("Invalid Input", "Class name cannot be empty.")
            return

        class_path = os.path.join(DATASET_DIR, class_name)
        if not os.path.exists(class_path):
            os.makedirs(class_path)
            messagebox.showinfo("Class Created", f"New class '{class_name}' was created.")

        upload_images(class_name)
        top.destroy()

    Button(top, text="Upload Images", command=on_confirm).pack(pady=10)


# Main
if __name__ == "__main__":
    create_ui()
