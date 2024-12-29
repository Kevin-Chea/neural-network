from PIL import Image

def pre_process_image(path):
    img = Image.open(path).convert('L')
    img_resized = img.resize((28, 28))
    img_resized.show()
    pixels = [pixel / 255.0 for pixel in img_resized.getdata()]
    return pixels

print(pre_process_image("data/by_class/30/hsf_0/hsf_0_00000.png"))