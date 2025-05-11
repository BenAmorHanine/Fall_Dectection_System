import numpy as np
import tensorflow as tf

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="snn_model_int8.tflite")

# Allocate tensors
interpreter.allocate_tensors()

# Get input details and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
###
input_scale, input_zero_point = input_details[0]['quantization']
###
# Create a dummy input with the correct shape
# Ensure the generated input has the correct dtype (float32)
"""dummy_input = np.random.random(input_details[0]['shape']).astype(np.float32)
dummy_input_quantized = np.round(dummy_input / input_scale + input_zero_point).astype(np.int8)
# Set the tensor
#interpreter.set_tensor(input_details[0]['index'], dummy_input)
interpreter.set_tensor(input_details[0]['index'], dummy_input_quantized)"""
####

accx=-1.0
accy=-246.0
accz=70.0
actual_input = np.array([[[accx], [accy], [accz]]], dtype=np.float32)  # Shape: [1, 3, 1]
actual_input_quantized = np.round(actual_input / input_scale + input_zero_point).astype(np.int8)
# Set the tensor
interpreter.set_tensor(input_details[0]['index'], actual_input_quantized)
# Run the model
interpreter.invoke()
# Get the output
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)

print(output_details)
# Assuming the output is a probability for binary classification
predicted_class = "Positive" if output_data[0][0] > 0.1 else "Negative"
print(f"Predicted class: {predicted_class}")


# Print input shape and data type
print("Input Shape:", input_details[0]['shape'])
print("Input Data Type:", input_details[0]['dtype'])

