import os
import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity as ssim
from datetime import datetime
import pickle
import uuid
from pprint import pprint
import time

UPLOADS_FOLDER = 'uploads'
THRESHOLD = 0.8
NEW_IMAGE_PATH = 'palm.jpg'
NUMPY_FOLDER = 'static/numpyArray'


def save_image_file(image, filename):
    cv.imwrite(filename, image)

def process_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    blurred = cv.GaussianBlur(equalized, (5, 5), 0)
    edges = cv.Canny(blurred, 30, 70)
    lined = np.zeros_like(gray)  # Changed to same shape as gray image
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 20, minLineLength=20, maxLineGap=5)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv.line(lined, (x1, y1), (x2, y2), 255, 1)  # Drawing white lines on black background
    return lined, lines

def process_and_save_new_image(image_path,saveImage=False):
    FILENAME = os.path.basename(image_path)
    image = cv.imread(image_path)
    palm_lines, lines = process_image(image)
    if saveImage:
        np.save(f"static/numpyArray/{FILENAME}", palm_lines)
        cv.imwrite(f'uploads/{FILENAME}_processed.jpg', palm_lines)

    return palm_lines
    # cv.imwrite(os.path.join(UPLOADS_FOLDER, f'{datetime.now().strftime("%Y%m%d%H%M%S")}_processed.jpg'), palm_lines)

def compare_with_saved_arrays(new_array):
    global NUMPY_FOLDER
    assert os.path.exists(NUMPY_FOLDER), "NUMPY_FOLDER is not Exist"

    try:
        similarities = []
        # - getting Files of Numpy
        files = os.listdir(NUMPY_FOLDER)
        if len(files) <= 0:
            raise AssertionError("No Numpy Files Found")

        for num in files:
            if num.endswith('.npy'):
                new_array_h, new_array_w = new_array.shape
                saved_array = np.load(os.path.join(NUMPY_FOLDER,num))
                # Check if saved_array is not empty and has valid dimensions
                if saved_array.size > 0 and len(saved_array.shape) == 2:
                    try:
                        saved_array_resized = cv.resize(saved_array, (new_array_w, new_array_h))
                        similarity_index = ssim(new_array, saved_array_resized)
                        similarities.append((num, similarity_index))
                    except cv.error as e:
                        print(f"Error resizing array {num}: {e}")
            else:
                print(f"{num} Is not Numpy File")

        return similarities        
    except Exception as e:
        print(f"Error: {e}")
        return []

def compare_with_saved_arrays_withpickle(new_array):
    global NUMPY_FOLDER
    assert os.path.exists(NUMPY_FOLDER), "NUMPY_FOLDER is not Exist"
    try:
        similarities = []
        # - Reading pkl file
        with open("Myfile.pkl",'rb') as f:
            data = pickle.load(f)

        for i in data:
            Filename = i['original_name']
            FileData = i['Data']
            new_array_h, new_array_w = new_array.shape

            if FileData.size > 0 and len(FileData.shape) == 2:
                try:
                    saved_array_resized = cv.resize(FileData, (new_array_w, new_array_h))
                    similarity_index = ssim(new_array, saved_array_resized)
                    similarities.append((Filename, similarity_index))
                except cv.error as e:
                    print(f"Error resizing array {Filename}: {e}")

        return similarities
    except Exception as e:
        print(f"Error: {e}")
        return []


def RegisterPalms(palmFolder):
    os.makedirs(f"{palmFolder}/Completed",exist_ok=True)
    # ` Registring some Palms
    for i in os.listdir(f"{palmFolder}"):
        if i.endswith(".jpg"):
            try:
                process_and_save_new_image(os.path.join(palmFolder,i),True)
                os.rename(os.path.join(palmFolder,i),os.path.join(palmFolder,"Completed",i))
                print("File moved!")
            except:
                continue       

if __name__ == "__main__":
    # ! Register Palms for Test
    # RegisterPalms('palms')
    start = time.time() 

    if not os.path.exists(NEW_IMAGE_PATH):
        print(f"Error: The file {NEW_IMAGE_PATH} does not exist.")
        exit(1)

    os.makedirs(UPLOADS_FOLDER, exist_ok=True)

    new_array = process_and_save_new_image(NEW_IMAGE_PATH)

    similarities = compare_with_saved_arrays(new_array)

    # sorting hight to low
    # similarities.sort(key=lambda x: x[1],reverse=True)
    # pprint(f"Matched {similarities[0][0]} with {similarities[0][1]}")


    for filename, similarity_index in similarities:
        print(f"Similarity with {filename}: {similarity_index:.3f}")

    if similarities:
        most_similar = max(similarities, key=lambda x: x[1])
        print(f"\nMost similar image: {most_similar[0]} with a similarity index of {most_similar[1]:.3f}")
        similarity_result = "The palms are similar." if most_similar[1] > THRESHOLD else "The palms are different."
        print(similarity_result)
    else:
        print("No saved arrays found for comparison.")

    print(f"Total time teken: {time.time() - start} sec")