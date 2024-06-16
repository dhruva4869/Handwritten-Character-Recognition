import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('handwriting.model')

def predict_image(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        
        if not black_bg_var.get():
            img = np.invert(img)
        
        img = np.array([img])
        img = img.reshape((1, 28, 28, 1))
        prediction = model.predict(img)
        number = np.argmax(prediction)
        return number
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def display_image(image_path, number):
    img = Image.open(image_path)
    img.thumbnail((300, 300)) 
    img = ImageTk.PhotoImage(img)
    canvas.delete("all")
    
    canvas.create_image(150, 150, image=img)
    canvas.image = img
    
    prediction_text = f"The number is probably a {number}" if number is not None else "Prediction failed!"
    label_prediction.config(text=prediction_text)


def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        number = predict_image(file_path)
        display_image(file_path, number)


def upload_folder():
    folder_path = filedialog.askdirectory()
    if folder_path:
        image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path)
                       if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if image_paths:
            i = 0

            def show_next_image():
                nonlocal i
                if i < len(image_paths):
                    image_path = image_paths[i]
                    number = predict_image(image_path)
                    display_image(image_path, number)
                    i += 1
                    root.after(5000, show_next_image)

            show_next_image() 

root = tk.Tk()
root.title("Digit Recognition")

# display image uploaded by user
canvas = tk.Canvas(root, width=300, height=300, bg='white')
canvas.pack()

# display the prediction
label_prediction = tk.Label(root, text="", font=('Helvetica', 16))
label_prediction.pack()

im_button = tk.Button(root, text="Upload Image", command=upload_image)
im_button.pack()

fol_button = tk.Button(root, text="Upload Folder", command=upload_folder)
fol_button.pack()

black_bg_var = tk.BooleanVar()
black_bg_checkbox = tk.Checkbutton(root, text="Does Image has a black background ?", variable=black_bg_var)
black_bg_checkbox.pack()

root.mainloop()
