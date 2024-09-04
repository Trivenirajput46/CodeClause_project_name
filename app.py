import os
import cv2 as cv
import numpy as np
import mediapipe as mp
from skimage.metrics import structural_similarity as ssim
from flask import Flask, render_template, request
import pickle
from datetime import datetime

THRESHOLD = 0.8

app = Flask(__name__,template_folder="../templates",static_folder='../static')

def save_image_file(image, filename):
    cv.imwrite(filename, image)

def process_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    blurred = cv.GaussianBlur(equalized, (5, 5), 0)
    edges = cv.Canny(blurred, 30, 70)
    lined = np.zeros_like(image)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 20, minLineLength=20, maxLineGap=5)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv.line(lined, (x1, y1), (x2, y2), (0, 255, 0), 1)
    output = cv.addWeighted(image, 0.8, lined, 1, 0)
    return edges, output, lines

def align_hand(image, hand_landmarks):
    coords = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark])
    center = np.mean(coords, axis=0)
    angle = np.arctan2(coords[-1, 1] - center[1], coords[-1, 0] - center[0])
    h, w = image.shape[:2]
    M = cv.getRotationMatrix2D(tuple(center * [w, h]), np.degrees(angle) - 90, 1)
    aligned_image = cv.warpAffine(image, M, (w, h))
    return aligned_image

def count_palm_lines(lines):
    if lines is not None:
        return len(lines)
    return 0

def process_and_compare_images(image1, image2, user1, user2):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    images = [image1, image2]
    palm_images = []
    processed_outputs = []
    line_counts = []

    for idx, image in enumerate(images):
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        result = hands.process(rgb_image)

        if result.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
                aligned_image = align_hand(image, hand_landmarks)
                aligned_rgb_image = cv.cvtColor(aligned_image, cv.COLOR_BGR2RGB)
                result_aligned = hands.process(aligned_rgb_image)

                if result_aligned.multi_hand_landmarks:
                    hand_landmarks_aligned = result_aligned.multi_hand_landmarks[0]
            
                    h, w, _ = aligned_image.shape
                    x_min = int(min([lm.x for lm in hand_landmarks_aligned.landmark]) * w)
                    x_max = int(max([lm.x for lm in hand_landmarks_aligned.landmark]) * w)
                    y_min = int(min([lm.y for lm in hand_landmarks_aligned.landmark]) * h)
                    y_max = int(max([lm.y for lm in hand_landmarks_aligned.landmark]) * h)
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(w, x_max)
                    y_max = min(h, y_max)
                    palm_region = aligned_image[y_min:y_max, x_min:x_max]
                    palm_edges, output_image, lines = process_image(palm_region)
                    palm_images.append(palm_edges)
                    processed_outputs.append(output_image)
                    line_count = count_palm_lines(lines)
                    line_counts.append(line_count)

                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    lines_filename = f'static/numpyArray/{user1 if idx == 0 else user2}_lines_{timestamp}.npy'
                    np.save(lines_filename, lines)
                    np.savetxt(f'text/{user1 if idx == 0 else user2}_lines_{timestamp}.txt', lines.reshape(-1, 4), fmt='%d')

    if len(palm_images) == 2:
        palm_images[0] = cv.resize(palm_images[0], (palm_images[1].shape[1], palm_images[1].shape[0]))
        similarity_index = ssim(palm_images[0], palm_images[1])
        similarity_index_str = f"{similarity_index:.3f}"
        np.save('static/numpyArray/similarity_metrics.npy', np.array([similarity_index]))
        similarity_result = "The palms are similar." if similarity_index > THRESHOLD else "The palms are different."

        output_image1_path = os.path.join("output", f"output_{user1}_{timestamp}.jpg")
        output_image2_path = os.path.join("output", f"output_{user2}_{timestamp}.jpg")
        save_image_file(processed_outputs[0], output_image1_path)
        save_image_file(processed_outputs[1], output_image2_path)

        matrix1 = palm_images[0]
        matrix2 = palm_images[1]
        matrix1_clean = matrix1[np.isfinite(matrix1)]
        matrix2_clean = matrix2[np.isfinite(matrix2)]
        data = {user1: matrix1_clean.tolist(), user2: matrix2_clean.tolist()}
        with open(f"text/matrix_data_{timestamp}.pkl", "wb") as f:
            pickle.dump(data, f)

        return similarity_index_str, similarity_result, line_counts
    else:
        return None, "Two palms were not detected.", []

@app.route("/", methods=["GET", "POST"])
def index():
    similarity_index = None
    similarity_result = None
    user1 = None
    user2 = None
    line_counts = []

    if request.method == "POST":
        user1 = request.form["user1"]
        user2 = request.form["user2"]
        image1 = request.files["image1"]
        image2 = request.files["image2"]

        if image1 and image2 and user1 and user2:
            image1_path = os.path.join("uploads", f"{user1}_image1_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
            image2_path = os.path.join("uploads", f"{user2}_image2_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
            image1.save(image1_path)
            image2.save(image2_path)
            image1_cv = cv.imread(image1_path)
            image2_cv = cv.imread(image2_path)
            similarity_index, similarity_result, line_counts = process_and_compare_images(image1_cv, image2_cv, user1, user2)

    return render_template("index.html", similarity_index=similarity_index, similarity_result=similarity_result, user1=user1, user2=user2, line_counts=line_counts)

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("static/numpyArray", exist_ok=True)
    os.makedirs("text", exist_ok=True)
    app.run(debug=True, host='0.0.0.0')
