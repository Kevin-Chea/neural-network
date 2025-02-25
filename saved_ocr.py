from neural_network import Neural_Network
from image_processing import pre_process_image, inverse_colors, image_base_path, folder_of_number

MODEL_TO__LOAD = "ocr_overtrained.pkl"

def predict_image_with_probability(nn, image_path):
    pixels = pre_process_image(image_path)
    inverted_pixels = inverse_colors(pixels)
    output = nn.forward(inverted_pixels)
    #print(output)
    predicted_label = output.index(max(output))  # Prend l'index avec la probabilité la plus haute
    return predicted_label, output, max(output)

def predict_all_numbers(nn):
    for i in range(10):
        result, output, probability = predict_image_with_probability(nn, 
                            image_base_path 
                            + folder_of_number(i) 
                            + "/train_3"+str(i)
                            + "/train_3"+str(i)+"_20000.png"
        )
        print(f"For {i}, the network predicts {result} with a probability of {probability}")
        

#nn = Neural_Network.load_model("MODEL_TO__LOAD")
#predict_all_numbers(nn)