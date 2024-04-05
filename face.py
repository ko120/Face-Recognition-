from PIL import Image
import face_recognition
import os
import pdb
from tqdm import tqdm  # Import tqdm

directory_path = "test"

save_directory = "test_extracted"
os.makedirs(save_directory, exist_ok=True)

files = os.listdir(directory_path)
total_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

for filename in tqdm(total_files, desc='Processing images'):
    image_path = os.path.join(directory_path, filename)

    if os.path.isfile(image_path):
        image = face_recognition.load_image_file(image_path)

        face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")


        if len(face_locations) > 1:
            # If more than one face is detected, skip this image
            face_locations = face_locations[0]
        elif len(face_locations) == 0:
            # If no faces are detected, also skip this image
            continue
        else:
            # If exactly one face is detected, proceed
            face_locations = face_locations[0]

        base_filename_without_ext = os.path.splitext(filename)[0]

        top, right, bottom, left = face_locations
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)

        face_filename = f"{base_filename_without_ext}.jpg"
        save_path = os.path.join(save_directory, face_filename)

        pil_image.save(save_path)

