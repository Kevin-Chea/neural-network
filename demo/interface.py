import tkinter as tk
from PIL import Image, ImageOps, ImageGrab, EpsImagePlugin, ImageDraw
import io

from neural_network import Neural_Network

nn = Neural_Network.load_model("result/ocrV2.pkl")


canvas_width, canvas_height = 200, 200
image1 = Image.new("L", (canvas_width, canvas_height), 'white')
draw = ImageDraw.Draw(image1)

window = tk.Tk()
canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg="white")
canvas.pack()

# --- Dessin ---
last_x, last_y = None, None

def draw_line(event):
    x, y = event.x, event.y
    canvas.create_line(x, y, x+8, y+8, fill='black', width=8)
    draw.ellipse((x, y, x+8, y+8), fill='black')

canvas.bind("<B1-Motion>", draw_line)

# --- Bouton de prédiction ---
def predict_digit(): 
    img = image1.resize((28, 28), Image.Resampling.LANCZOS)
    img = ImageOps.invert(img)
    # img.show()
    
    # Convertir en niveaux de gris, redimensionner, inverser les couleurs
    threshold = 0.2
    pixels = [(pixel / 255.0) for pixel in img.getdata()]  # inverser déjà ici
    pixels = [1.0 if p > threshold else 0.0 for p in pixels]
    
    # Prediction
    output = nn.forward(pixels)
    print(output)
    
btn = tk.Button(window, text="Prédire", command=predict_digit)
btn.pack()

# --- Bouton reset ---
def clear_canvas():
    canvas.delete("all")
    global image1, draw
    image1 = Image.new("L", (canvas_width, canvas_height), 'white')  # Image noire
    draw = ImageDraw.Draw(image1)

btn_clear = tk.Button(window, text="Effacer", command=clear_canvas)
btn_clear.pack(pady=5)

window.mainloop()