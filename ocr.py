from neural_network import Neural_Network
from image_processing import load_batch_for_all_numbers, pre_process_image, inverse_colors, image_base_path, folder_of_number
from math_utils import softmax

def build_ocr_network():
    nn = Neural_Network()
    nn.add_layer(128, 784)
    nn.add_layer(64, 128)
    nn.add_layer(10, 64, softmax, None) # 10 values, since it is only numbers for now (from 0 to 9)
    return nn

nn = build_ocr_network()
nn.train_with_batch(load_batch_for_all_numbers, 5, 10, 0.1)

def predict_image(nn, image_path):
    pixels = pre_process_image(image_path)
    inverted_pixels = inverse_colors(pixels)
    output = nn.forward(inverted_pixels)
    predicted_label = output.index(max(output))  # Prend l'index avec la probabilité la plus haute
    return predicted_label

def predict_all_numbers(nn):
    for i in range(10):
        result = predict_image(nn, 
                            image_base_path 
                            + folder_of_number(i) 
                            + "/train_3"+str(i)
                            + "/train_3"+str(i)+"_00000.png"
        )
        print(f"For {i}, the network predicts {result}")

predict_all_numbers(nn)

# If you want to save the model, uncomment the following line and adapt the filepath if needed
nn.save_model('ocrV2.pkl')
