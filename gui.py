import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from test import predict_image
import matplotlib.pyplot as plt
from wiki_info import get_bird_info

class BirdClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Bird Classifier")
        self.root.geometry('1200x600')

        self.root.configure(background='#36393f')

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        # Background gradient
        self.gradient_canvas = tk.Canvas(root, width=1200, height=600, bd=0, highlightthickness=0)
        self.gradient_canvas.place(x=0, y=0)

        self.gradient_canvas.create_rectangle(0, 0, 1200, 600, fill="#5865f2", outline="")

        self.gradient_canvas.create_rectangle(0, 0, 1200, 10, fill="#5865f2", outline="")
        for i in range(60):
            self.gradient_canvas.create_rectangle(0, 10 + i * 10, 1200, 20 + i * 10, fill="#36393f", outline="")

        self.gradient_canvas.create_rectangle(0, 610, 1200, 600, fill="#36393f", outline="")
        for i in range(60):
            self.gradient_canvas.create_rectangle(0, 600 - i * 10, 1200, 590 - i * 10, fill="#5865f2", outline="")

        # Left side of the window for input
        self.upload_button = tk.Button(root, text="Upload an image", command=self.load_image,
                                       font=("Helvetica", 14), bg='#5865f2', fg='white', bd=0, padx=20, pady=10)
        self.upload_button.place(x=20, y=20)

        self.image_label = tk.Label(root, bd=5, relief='solid', bg='#36393f')
        self.image_label.place(x=30, y=100, width=400, height=400)

        # Right side of the window for output
        self.result_image_label = tk.Label(root, bd=5, relief='solid', bg='#36393f')
        self.result_image_label.place(x=770, y=100, width=400, height=400)

        self.result_label = tk.Label(root, text="Predicted Bird", font=("Helvetica", 24, 'bold'), bg='#36393f', fg='white')
        self.result_label.place(x=480, y=180)

        self.info_frame = tk.Frame(root, bd=0, bg='#36393f')
        self.info_frame.place(x=480, y=250, width=280, height=250)

        self.info_canvas = tk.Canvas(self.info_frame, bd=0, highlightthickness=0, bg='#36393f')
        self.info_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.info_scrollbar = tk.Scrollbar(self.info_frame, orient=tk.VERTICAL, command=self.info_canvas.yview,
                                           bg='#36393f', troughcolor='#2f3136')
        self.info_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.info_canvas.configure(yscrollcommand=self.info_scrollbar.set)
        self.info_canvas.bind('<Configure>', self.configure_info_canvas)

        self.info_inner_frame = tk.Frame(self.info_canvas, bg='#36393f')
        self.info_canvas.create_window((0, 0), window=self.info_inner_frame, anchor=tk.NW)

    def load_image(self):
        file_path = filedialog.askopenfilename()
        self.image = Image.open(file_path)
        self.display_image(self.image, self.image_label)
        prediction = predict_image(file_path)
        self.result_label.config(text=f"Predicted Bird: {prediction}")

        # Clear previous bird information
        for widget in self.info_inner_frame.winfo_children():
            widget.destroy()

        # Fetch bird information from Wikipedia
        bird_info = get_bird_info(prediction)
        if bird_info == "None":
            bird_info = "Information not found."
        self.info_label = tk.Label(self.info_inner_frame, text=f"Bird Information: {bird_info}",
                                   font=("Helvetica", 14), bg='#36393f', fg='white', wraplength=250,
                                   justify='left')
        self.info_label.pack(pady=10, padx=10, anchor='w')  # Display bird information

        # Display the first bird image of the predicted class
        predicted_bird_image_path = f"C:\\BIRDFINAL\\archive\\train\\{prediction.upper()}\\001.jpg"
        predicted_bird_image = Image.open(predicted_bird_image_path)
        self.display_image(predicted_bird_image, self.result_image_label)

    def display_image(self, image, label_widget):
        image = image.resize((400, 400), Image.LANCZOS)

        photo = ImageTk.PhotoImage(image)
        label_widget.config(image=photo)
        label_widget.image = photo

    def configure_info_canvas(self, event):
        self.info_canvas.configure(scrollregion=self.info_canvas.bbox("all"))

    def on_mousewheel(self, event):
        self.info_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

if __name__ == "__main__":
    root = tk.Tk()
    gui = BirdClassifierGUI(root)
    root.bind_all("<MouseWheel>", gui.on_mousewheel)
    root.mainloop()
