from neural_network import Neural_Network
from saved_ocr import predict_image_with_probability
from image_processing import image_base_path, folder_of_number

#MODEL_TO_LOAD = "ocr_overtrained.pkl"
MODEL_BASE_PATH = "result/"
MODEL_TO_LOAD = "ocrV2.pkl"

def pad_number(number: int, length: int) -> str:
    return str(number).zfill(length)

def predict_all_trained(nn, nb_pictures=100):
    print("BEGIN PREDICTION ON TRAIN DATA")
    for n in range(10):
        nbErrors = 0
        for i in range(nb_pictures):
            result, output, probability = predict_image_with_probability(nn, 
                image_base_path 
                + folder_of_number(n) 
                + "/train_3"+str(n)
                + "/train_3"+str(n)
                + "_"
                + pad_number(i, 5)
                + ".png"
            )
            if result != n:
                nbErrors += 1
        print(f"For {n} the network failed {nbErrors} times out of {nb_pictures} ({(nbErrors / nb_pictures) * 100}%)")
    print("END PREDICTION ON TRAIN DATA")
        
def predict_not_trained(nn, nb_pictures=500):
    print("BEGIN PREDICTION ON NOT TRAIN DATA")
    for n in range(10):
        nbErrors = 0
        for i in range(500):
            result, output, probability = predict_image_with_probability(nn, 
                image_base_path 
                + folder_of_number(n) 
                + "/train_3"+str(n)
                + "/train_3"+str(n)
                + "_2"
                + pad_number(i, 4)
                + ".png"
            )
            if result != n:
                nbErrors += 1
        print(f"For {n} the network failed {nbErrors} times out of {nb_pictures} ({(nbErrors / nb_pictures) * 100}%)")
    print("END PREDICTION ON NOT TRAIN DATA")    
    
def load_model(model_to_load):
    return Neural_Network.load_model(MODEL_BASE_PATH + model_to_load)

#nn = Neural_Network.load_model(MODEL_TO_LOAD)
#predict_all_trained(nn, 1000)
#predict_all_trained(nn)
#print("")
#predict_not_trained(nn)