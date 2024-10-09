import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model


# Load the pre-trained MobileNetV2 model, excluding the top layers
class_names = ['dislike', 'like', 'stop_inverted', 'ok', 'mute', 'two_up', 'stop', 'peace', 'two_up_inverted', 'three', 'three2', 'call', 'one', 'rock', 'peace_inverted', 'fist', 'palm', 'four']
dataset_dir = "dataset/"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPUs Available: ", len(gpus))
    except RuntimeError as e:
        print(e)
# Load the dataset and split into training and validation sets
dataset = image_dataset_from_directory(
    dataset_dir,
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical'
)

# Split the dataset into training and validation sets (e.g., 80% train, 20% test)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset = dataset.take(train_size)
test_dataset = dataset.skip(train_size)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
predictions = Dense(len(class_names), activation='softmax')(x)

# Create the full model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model to avoid training them
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Train the model
model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=100
)

# Optionally, unfreeze some layers of the base model and fine-tune
for layer in base_model.layers[-20:]:  # Unfreeze last 20 layers
    layer.trainable = True

# Re-compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2), loss='categorical_crossentropy', metrics=['accuracy'])

# Fine-tune the model
model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=100
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_dataset)
print(f'Test Accuracy: {test_acc:.2f}')

# Save the trained model
model.save('hand_gesture_mobilenet.h5')