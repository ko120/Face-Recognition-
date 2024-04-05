from PIL import Image
import face_recognition
import os
import pdb
from tqdm import tqdm  # Import tqdm

# Specify the path to your directory containing images
directory_path = "test"

# Specify the directory where you want to save the face images
save_directory = "test_extracted"
# Ensure the save directory exists
os.makedirs(save_directory, exist_ok=True)

# Get the list of files in the directory and wrap it with tqdm for the progress bar
files = os.listdir(directory_path)
total_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Iterate over every image in the directory with a progress bar
for filename in tqdm(total_files, desc='Processing images'):
    # Construct the path to the image
    image_path = os.path.join(directory_path, filename)

    # Check if the file is an image (by extension)
    if os.path.isfile(image_path):
        # Load the image into a numpy array
        image = face_recognition.load_image_file(image_path)

        # Find all faces in the image using the CNN model
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

        # Extract the base name of the file without the extension
        base_filename_without_ext = os.path.splitext(filename)[0]

        # Extract each face from the image
        top, right, bottom, left = face_locations
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)

        # Construct a new filename for each extracted face
        face_filename = f"{base_filename_without_ext}.jpg"
        save_path = os.path.join(save_directory, face_filename)

        # Save the face image to disk
        pil_image.save(save_path)

