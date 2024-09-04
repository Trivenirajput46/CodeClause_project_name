import os
import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity as ssim
import time

data_folder='DATA'
collected_data='static/numpyArray'
new_collected_data = 'static/newNumpyArray'

#creating numpy array
def processing_imAGE(image_path):
    image=cv.imread(image_path)
    gray_image=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    resized_image=cv.resize(gray_image,(100,100))
    return resized_image

existing_arrays = []
existing_array_files = []
start = time.time() 

for filename in os.listdir(collected_data):
    if filename.endswith('.npy'):
        array_path = os.path.join(collected_data, filename)
        array = np.load(array_path)
        existing_arrays.append(array)
        existing_array_files.append(filename)
        # print("helloo222")

for image_filename in os.listdir(data_folder):
    #  print(image_filename)
     if image_filename.endswith(('.JPG', '.jpg')):
        image_path = os.path.join(data_folder, image_filename)
        new_image_array = processing_imAGE(image_path)
        # print("hello")

        # saving array
        new_array_filename = f"{os.path.splitext(image_filename)[0]}.npy"
        new_array_path = os.path.join(new_collected_data, new_array_filename)
        np.save(new_array_path, new_image_array)

        print(f"\nComparing {image_filename} with existing arrays:")

        highest_similarity = -1
        most_similar_file = None

        for i, existing_array in enumerate(existing_arrays):
            resized_existing_array = cv.resize(existing_array, (100, 100))
            similarity_index = ssim(new_image_array, resized_existing_array)
            print(f"Similarity with {existing_array_files[i]}: {similarity_index:.3f}")

            if similarity_index > highest_similarity:
                highest_similarity = similarity_index
                most_similar_file = existing_array_files[i]

        print(f"Most similar to: {most_similar_file} with similarity index: {highest_similarity:.3f}")

print("\nComparison complete.")
print(f"Total time teken: {time.time() - start} sec")





















