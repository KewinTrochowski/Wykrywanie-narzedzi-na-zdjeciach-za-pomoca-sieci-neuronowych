from data_loader import load_data, DataGenerator
from model import create_model
from config import TRAIN_DATA_PATH, VAL_DATA_PATH, MODEL_SAVE_PATH, IMAGE_DIMS, EPOCHS

# Ustawienia i uruchomienie treningu
train_generator = DataGenerator(load_data(TRAIN_DATA_PATH))
val_generator = DataGenerator(load_data(VAL_DATA_PATH))

model = create_model(IMAGE_DIMS)
history = model.fit_generator(
    generator=train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator)
)

model.save(MODEL_SAVE_PATH)
