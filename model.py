from keras import Sequential
from tensorflow.keras import Conv2D, MaxPooling2D, Flatten, Dense


def create_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(4, activation='sigmoid')  # 4 output units for bounding box [x_min, y_min, x_max, y_max]
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model


model = create_model((224, 224, 3))
