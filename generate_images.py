import glob
import random
from PIL import Image


# Zaktualizowana funkcja do wklejania obiektu, która zwraca bounding box
def paste_object(bg_image, obj_image, angle):
    obj_rotated = obj_image.rotate(angle, expand=True)
    max_x = bg_image.width - obj_rotated.width
    max_y = bg_image.height - obj_rotated.height
    pos_x = random.randint(0, max_x)
    pos_y = random.randint(0, max_y)
    bg_image.paste(obj_rotated, (pos_x, pos_y), obj_rotated)
    return [pos_x, pos_y, pos_x + obj_rotated.width, pos_y + obj_rotated.height]


# Zmienne do przechowywania danych
data = []

obj_files = glob.glob('objects/*/*')
bg_files = glob.glob('backgrounds/*')

# Ustawienie liczby obrazów do wygenerowania
total_images = 20
selected_bg_files = random.choices(bg_files, k=total_images)

# Generowanie obrazów
for i, bg_path in enumerate(selected_bg_files):
    with Image.open(bg_path) as bg_image:
        num_objects = random.randint(2, 5)
        annotations = []
        for _ in range(num_objects):
            obj_path = random.choice(obj_files)
            with Image.open(obj_path) as obj_image:
                angle = random.uniform(0, 360)
                bbox = paste_object(bg_image, obj_image, angle)
                annotations.append(bbox + [obj_files.index(obj_path)])  # Dodanie identyfikatora klasy

        # Zapisanie zmodyfikowanego obrazu tła
        output_filename = f"output/modified_{i}.png"
        bg_image.save(output_filename)
        data.append((output_filename, annotations))

# Zapis danych do pliku CSV lub JSON
import json

with open('data.json', 'w') as f:
    json.dump(data, f)
