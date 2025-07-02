import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import cv2
import os 
import glob
from pathlib import Path
import multiprocessing
import sys
import io

nnfs.init()

IMAGE_SIZE = (64, 64)
NUM_CHANNELS = 3
INPUT_FEATURES = IMAGE_SIZE[0] * IMAGE_SIZE[1] * NUM_CHANNELS

CURRENT_PATH = Path(__file__).resolve()
DATASET_PATH = CURRENT_PATH.parent.parent / 'dogcat'
CLASSES = ['cats', 'dogs']
LABEL_MAP = {cls: i for i, cls in enumerate(CLASSES)}

def _process_single_image(args):
    img_path, label_id, image_size, num_channels = args

    # --- Redirect stderr (Standard Error) ---
    # Store the original stderr so we can restore it later
    original_stderr = sys.stderr
    # Redirect stderr to a dummy in-memory stream (StringIO)
    # This will capture any output to stderr, effectively silencing it from the console
    sys.stderr = io.StringIO()
    # --- End Redirect ---

    try:
        img = cv2.imread(img_path)
        if img is None:
            return None # Indicate failure

        if num_channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif num_channels == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, image_size)
        img = img.astype(np.float32) / 255.0
        img_flat = img.reshape(-1)

        return img_flat, label_id

    except Exception as e:
        return None # Indicate a processing error
    finally:
        # --- Restore stderr ---
        # IMPORTANT: Always restore stderr in a finally block to ensure it happens
        # even if an error occurs during image processing.
        sys.stderr = original_stderr
        # --- End Restore ---

def load_and_preprocess_images(data_dir, image_size, num_channels, label_map):
    all_image_args = [] # Will store (img_path, label_id, image_size, num_channels) tuples

    print(f"Collecting image paths from: {data_dir}")
    for class_name, class_label in label_map.items():
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            print(f"Warning: Class path not found: {class_path}. Skipping.")
            continue

        image_files = glob.glob(os.path.join(class_path, '*.jpg'))

        print(f"  Found {len(image_files)} images for class '{class_name}' at {class_path}")

        for img_path in image_files:
            # Store all necessary arguments for _process_single_image
            all_image_args.append((img_path, class_label, image_size, num_channels))

    print(f"Total images identified for processing: {len(all_image_args)}")

    # Determine the number of processes to use (usually CPU count is a good start)
    # Using os.cpu_count() provides the number of CPUs in the system
    num_processes = os.cpu_count() if os.cpu_count() else 4 # Fallback to 4 if cannot determine
    print(f"Using {num_processes} processes for parallel image loading and preprocessing...")

    # Create a multiprocessing Pool
    # 'starmap' is used because _process_single_image expects multiple arguments unpacked from a tuple
    with multiprocessing.Pool(processes=num_processes) as pool:
        processed_results = list(pool.map(_process_single_image, all_image_args))
        # Note: If your _process_single_image returns multiple values, you'd use pool.starmap
        # pool.map works when _process_single_image takes a single tuple argument, which is how we set it up.


    # Filter out any images that failed to load or process
    all_images = []
    all_labels = []
    failed_count = 0
    for result in processed_results:
        if result is not None:
            all_images.append(result[0])
            all_labels.append(result[1])
        else:
            failed_count += 1

    if failed_count > 0:
        print(f"Warning: {failed_count} images failed to load or process.")

    # Check if any images were loaded at all
    if not all_images:
        print("Error: No images were loaded successfully. Returning empty arrays.")
        dummy_flat_dim = image_size[0] * image_size[1] * num_channels
        return np.empty((0, dummy_flat_dim), dtype=np.float32), np.empty((0,), dtype=np.int32)


    # Convert lists to NumPy arrays
    X_data = np.array(all_images)
    y_data = np.array(all_labels)

    # Shuffle the dataset (important for training)
    permutation = np.random.permutation(len(X_data))
    X_data = X_data[permutation]
    y_data = y_data[permutation]

    return X_data, y_data

#Rectified Linear Object
class Activation_ReLU():
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, self.inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

#Softamx Object
class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(self.inputs - np.max(self.inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
    def backward(self, dvalues):
        self.dinputs = dvalues

#Loss object
class Loss:
    def caclulate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
     
#Loss (Using categorical crossentropy) Object
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 1:
            y_true_one_hot = np.zeros_like(dvalues)
            y_true_one_hot[range(samples), y_true] = 1
        else :
            y_true_one_hot = y_true

        self.dinputs = (dvalues - y_true_one_hot) / samples

#Layer object class
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(self.inputs, self.weights) + self.biases
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

#Optimize Object - Using the "dweights, dbiases" to update the weights and biases of Layer Dense Objects
class Optimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases

if __name__ == "__main__":
    X_train, y_train = load_and_preprocess_images(os.path.join(DATASET_PATH, 'train'), IMAGE_SIZE, NUM_CHANNELS, LABEL_MAP)
    X, y = X_train, y_train # Use your loaded image data

    dense1 = Layer_Dense(INPUT_FEATURES, 128)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(128, len(CLASSES))
    activation2 = Activation_Softmax()
    optimizer = Optimizer(0.003)
    loss_function = Loss_CategoricalCrossentropy()

    training = False
    testing = True

    if training == True:

        #Start of Training Loop
        num_epochs = 25
        batch_size = 32

        print(f"\nStarting training for {num_epochs} epochs with batch size {batch_size}...")

        for epoch in range(num_epochs):
            permutation = np.random.permutation(len(X_train))
            X_shuffled = X_train[permutation]
            y_shuffled = y_train[permutation]

            for i in range(0, len(X_train), batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                # 1. Forward Pass
                dense1.forward(X)
                activation1.forward(dense1.output)

                dense2.forward(activation1.output)
                activation2.forward(dense2.output)

                # 2. Loss Calculation
                loss = loss_function.caclulate(activation2.output, y)

                # 3. Backward Pass
                loss_function.backward(activation2.output, y)

                activation2.backward(loss_function.dinputs)
                dense2.backward(activation2.dinputs)

                activation1.backward(dense2.dinputs)
                dense1.backward(activation1.dinputs)

                # 4. Parameter update
                optimizer.update_params(dense1)
                optimizer.update_params(dense2)


            dense1.forward(X)
            activation1.forward(dense1.output)
            dense2.forward(activation1.output)
            activation2.forward(dense2.output)
            epoch_loss = loss_function.caclulate(activation2.output, y)

            predictions = np.argmax(activation2.output, axis=1)
            accuracy = np.mean(predictions == y_train) * 100

            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2}%")

        print("\n--- Training Complete ---")
