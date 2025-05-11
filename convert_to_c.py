import numpy as np

# Load the TFLite model file
with open("snn_model_int8.tflite", "rb") as f:
    model_data = f.read()
# Convert to a C array
c_array = ', '.join(f'0x{byte:02x}' for byte in model_data)
# Write to a .h file
with open("model_snn_fall_prediction.h", "w") as f:
    f.write("#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n")
    f.write(f"unsigned char model_data[] = {{ {c_array} }};\n")
    f.write(f"unsigned int model_data_len = {len(model_data)};\n\n")
    f.write("#endif // MODEL_DATA_H\n")

print("model_snn_fall_prediction.h file has been created.")
