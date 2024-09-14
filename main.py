import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import cv2
# Tải mô hình đã huấn luyện
model = tf.keras.models.load_model('model_pretrained1.h5')

def preprocess_image(img):

    img = cv2.resize(img,(224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def open_image():

    filepath = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])


    if not filepath:
        return

    # Lấy tên file từ đường dẫn
    true_label = filepath.split("/")[-1].split(".")[0]


    img = cv2.imread(filepath)
    img = cv2.resize(img,(400, 400))
    # Chuyển đổi từ BGR sang RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Chuyển đổi từ định dạng OpenCV sang định dạng mà Tkinter có thể hiển thị
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)

    # Hiển thị hình ảnh lên canvas
    canvas.create_image(0, 0, anchor="nw", image=img_tk)
    canvas.image = img_tk


    img_array = preprocess_image(img)

    # Dự đoán kết quả
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=-1)[0]
    confidence = round(100 * (np.max(predictions[0])), 2)


    class_labels = ["Audi","Hyundai Creta","Mahindra Scorpio", "Rolls Royce","Swift","Tata Safari",  "Toyota Innova"]
    result_text = f"Dự đoán: {class_labels[predicted_class]} \n Độ tin cậy: {confidence}%"
    result_label.config(text=result_text)

    # Hiển thị true label dưới hình ảnh
    true_label_text = f"{true_label}"
    true_label_widget.config(text=true_label_text)


# Tạo cửa sổ chính
root = tk.Tk()
root.title("Image Viewer")
root.configure(bg="#f0f0f0")


left_frame = tk.Frame(root, width=620, height=620, bg="#ffffff")
left_frame.pack(side="left", padx=10, pady=10)
left_frame.pack_propagate(False)


right_frame = tk.Frame(root, width=500, height=620, bg="#ffffff")
right_frame.pack(side="right", padx=10, pady=10)
right_frame.pack_propagate(False)

open_button = tk.Button(left_frame, text="Open Image", command=open_image,
                        bg="#4CAF50", fg="white", activebackground="#45a049", activeforeground="white")
open_button.pack(pady=10)


canvas = tk.Canvas(left_frame, width=400, height=400, bg="#e0e0e0")
canvas.pack()

# Thêm nhãn true label dưới canvas
true_label_widget = tk.Label(left_frame, text="None", bg="#ffffff", font=('Arial', 14))
true_label_widget.pack(pady=10)


result_label = tk.Label(right_frame, text="Dự đoán: None", wraplength=180, bg="#ffffff", font=('Arial', 14))
result_label.pack(expand=True, anchor='center', pady=10)


root.mainloop()
