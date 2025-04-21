from result.success_rate import load_model, predict_all_trained, predict_not_trained

MODEL_TO_LOAD = "ocrV2.pkl"

nn = load_model(MODEL_TO_LOAD)

predict_all_trained(nn, 1000)
print("")
predict_not_trained(nn)

# To execute this file, go to the root folder and type this command in your terminal : 
# python3 -m demo.ocr_demo_v2