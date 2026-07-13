"""
Welcome to BHAVIKA'S Digit Detective 🕵️
Draw a digit on the board and it will be guessed by ou ANN that is trained on MNIST dataset
"""

import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageTk
import numpy as np

# Loading the saved model 

MODEL_PATH = "mnist_model.h5"  
DRAW_CANVAS_SIZE = 240          

try:
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Could not load model at startup: {e}")


def preprocess_image(pil_img: Image.Image):
    """
    Convert the drawn digit into the 28x28 grayscale, normalized,
    MNIST-style format the model expects: shape (1, 28, 28, 1).
    Returns (model_input_array, debug_28x28_PIL_image).
    """
    img = pil_img.convert("L")
    np_img = np.array(img)

    # Threshold-based bounding box: find the actual digit strokes
    threshold = max(30, np_img.mean() + np_img.std())
    mask = np_img > threshold
    if mask.any():
        ys, xs = np.where(mask)
        top, bottom = ys.min(), ys.max()
        left, right = xs.min(), xs.max()
        img = img.crop((left, top, right + 1, bottom + 1))

    # Resize keeping aspect ratio into a 20x20 box (leaves a small margin,
    # same as the original MNIST normalization)
    img.thumbnail((20, 20), Image.LANCZOS)

    # Paste onto a 28x28 canvas, centered by center-of-mass — this is
    # closer to how real MNIST digits were centered and noticeably
    # improves accuracy on hand-drawn input
    canvas = Image.new("L", (28, 28), color=0)
    arr_small = np.array(img).astype("float32")

    if arr_small.sum() > 0:
        ys, xs = np.indices(arr_small.shape)
        total = arr_small.sum()
        cy = (ys * arr_small).sum() / total
        cx = (xs * arr_small).sum() / total
        offset_x = int(round(14 - cx))
        offset_y = int(round(14 - cy))
    else:
        offset_x = (28 - img.width) // 2
        offset_y = (28 - img.height) // 2

    canvas.paste(img, (offset_x, offset_y))

    arr = np.array(canvas).astype("float32") / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return arr, canvas


class DigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BHAVIKA'S Digit Detective 🕵️ ")
        self.root.minsize(520, 460)
        self.root.geometry("560x520")

        # Make the layout stretch with the window
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(1, weight=1)

        # Heading 
        self.heading_label = tk.Label(root, text="BHAVIKA'S Digit Detective 🕵️ ",
                                       font=("Arial", 18, "bold"), fg="blue")
        self.heading_label.grid(row=0, column=0, columnspan=2, pady=(15, 5), sticky="ew")

        # frame to draw the digit 
        left_frame = tk.Frame(root, padx=10, pady=10)
        left_frame.grid(row=1, column=0, sticky="nsew")

        tk.Label(left_frame, text="Draw a Digit", font=("Arial", 12, "bold")).pack()
        self.draw_canvas = tk.Canvas(left_frame, width=DRAW_CANVAS_SIZE, height=DRAW_CANVAS_SIZE,
                                      bg="black", cursor="cross",
                                      highlightthickness=1, highlightbackground="gray")
        self.draw_canvas.pack(pady=5, expand=True)
        self.draw_canvas.bind("<B1-Motion>", self.paint)

        self.draw_image = Image.new("L", (DRAW_CANVAS_SIZE, DRAW_CANVAS_SIZE), color=0)
        self.draw_draw = ImageDraw.Draw(self.draw_image)

        btn_row = tk.Frame(left_frame)
        btn_row.pack(pady=5, fill="x")
        tk.Button(btn_row, text="Clear", command=self.clear_canvas).pack(side="left", expand=True, fill="x")
        tk.Button(btn_row, text="Predict", command=self.predict_drawing).pack(side="left", expand=True, fill="x")

        # frame to diplay what the model sees after preprocessing
        right_frame = tk.Frame(root, padx=10, pady=10)
        right_frame.grid(row=1, column=1, sticky="nsew")

        tk.Label(right_frame, text="What the Model Sees (28x28)", font=("Arial", 12, "bold")).pack()
        self.canvas_debug = tk.Canvas(right_frame, width=200, height=200, bg="black",
                                       highlightthickness=1, highlightbackground="gray")
        self.canvas_debug.pack(pady=5, expand=True)
        tk.Label(right_frame, text="If this looks wrong/unclear,\nthe prediction will be too.",
                 font=("Arial", 9), fg="gray").pack()

        # Result
        self.result_label = tk.Label(root, text="Prediction: -", font=("Arial", 16, "bold"), fg="blue")
        self.result_label.grid(row=2, column=0, columnspan=2, pady=10, sticky="ew")

        self.tk_debug_img = None

    # draw the digit
    def paint(self, event):
        r = 9
        x, y = event.x, event.y
        self.draw_canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self.draw_draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    # to clear the canvas
    def clear_canvas(self):
        self.draw_canvas.delete("all")
        self.draw_image = Image.new("L", (DRAW_CANVAS_SIZE, DRAW_CANVAS_SIZE), color=0)
        self.draw_draw = ImageDraw.Draw(self.draw_image)
        self.result_label.config(text="Prediction: -")
        self.canvas_debug.delete("all")

    # to predict the drawn image
    def predict_drawing(self):
        self.run_prediction(self.draw_image)

    # Prediction logic 
    def run_prediction(self, pil_img):
        if model is None:
            messagebox.showerror(
                "Model not loaded",
                f"Could not load model from '{MODEL_PATH}'. "
                "Update MODEL_PATH in the script to point to your saved model."
            )
            return

        arr, processed = preprocess_image(pil_img)

        # Show the debug 28x28 image, scaled up so it's actually visible
        debug_display = processed.resize((200, 200), Image.NEAREST)
        self.tk_debug_img = ImageTk.PhotoImage(debug_display)
        self.canvas_debug.delete("all")
        self.canvas_debug.create_image(0, 0, anchor="nw", image=self.tk_debug_img)

        preds = model.predict(arr, verbose=0)[0]
        digit = int(np.argmax(preds))
        self.result_label.config(text=f"Prediction: {digit}")


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitApp(root)
    root.mainloop()