# import torch
# checkpoint = torch.load('C:/Users/garla/OneDrive/Desktop/cipher-yurei/app/sign.pt', map_location='cpu')
# print(f"Type: {type(checkpoint)}")
# if isinstance(checkpoint, dict):
#     print(f"Keys: {checkpoint.keys()}")
# else:
#     print(f"Model class: {checkpoint.__class__.__name__}")


import numpy as np
from tensorflow import keras

# Load your model
model = keras.models.load_model('C:/Users/garla/Downloads/action.h5')

# Check model info
print("=" * 50)
print("MODEL INFORMATION:")
print("=" * 50)
print(f"Input shape: {model.input_shape}")
print(f"Output shape: {model.output_shape}")
print(f"Number of classes: {model.output_shape[-1]}")
print("\nModel Summary:")
model.summary()

# Test with random data
print("\n" + "=" * 50)
print("TESTING WITH RANDOM DATA:")
print("=" * 50)
sequence_length = model.input_shape[1]
num_features = model.input_shape[2]

test_sequence = np.random.rand(1, sequence_length, num_features)
predictions = model.predict(test_sequence)

print(f"Predictions: {predictions}")
print(f"Predicted class: {np.argmax(predictions)}")
print(f"Confidence: {np.max(predictions):.4f}")