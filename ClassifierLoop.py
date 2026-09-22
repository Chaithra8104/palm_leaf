from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.transform import resize
from joblib import load
import time

class_names_base = ['ಅ', 'ಆ', 'ಇ', 'ಈ', 'ಉ', 'ಊ', 'ಋ', 'ಎ', 'ಏ', 'ಐ', 'ಒ', 'ಓ', 'ಔ', 'ಕ', 'ಖ', 'ಗ', 'ಘ', 'ಚ', 'ಛ', 'ಜ', 'ಟ', 'ಡ', 'ಣ', 'ತ', 'ಥ', 'ದ', 'ಧ', 'ನ', 'ಪ', 'ಫ', 'ಬ', 'ಭ', 'ಮ', 'ಯ', 'ರ', 'ಲ', 'ಳ', 'ವ', 'ಶ', 'ಷ', 'ಸ', 'ಹ']

class_names_mod = ['ಅ', 'ಅಂ', 'ಅಃ', 'ಆ', 'ಇ', 'ಈ', 'ಉ', 'ಊ', 'ಋ', 'ಎ', 'ಏ', 'ಐ', 'ಒ', 'ಓ']

class_names_ott = ['Null', 'ಕ', 'ಖ', 'ಗ', 'ಚ', 'ಞ', 'ಟ', 'ಠ', 'ಣ', 'ತ', 'ಥ', 'ದ', 'ನ', 'ಪ', 'ಬ', 'ಮ', 'ಯ', 'ರ', 'ರ್', 'ಲ', 'ವ', 'ಷ']


kannada_unicode = {
    "ಅ": "\u0C85", 
    "ಆ": "\u0C86", 
    "ಇ": "\u0C87", 
    "ಈ": "\u0C88",
    "ಉ": "\u0C89",
    "ಊ": "\u0C8A",
    "ಋ": "\u0C8B", 
    "ೠ": "\u0CE0", 
    "ಎ": "\u0C8E",
    "ಏ": "\u0C8F",
    "ಐ": "\u0C90", 
    "ಒ": "\u0C92",
    "ಓ": "\u0C93", 
    "ಔ": "\u0C94",
    "mಅಂ": "\u0C82",
    "mಅಃ": "\u0C83",
    "ಕ": "\u0C95", 
    "ಖ": "\u0C96", 
    "ಗ": "\u0C97", 
    "ಘ": "\u0C98",
    "ಙ": "\u0C99", 
    "ಚ": "\u0C9A",
    "ಛ": "\u0C9B", 
    "ಜ": "\u0C9C", 
    "ಝ": "\u0C9D",
    "ಞ": "\u0C9E", 
    "ಟ": "\u0C9F", 
    "ಠ": "\u0CA0", 
    "ಡ": "\u0CA1", 
    "ಢ": "\u0CA2",
    "ಣ": "\u0CA3", 
    "ತ": "\u0CA4", 
    "ಥ": "\u0CA5", 
    "ದ": "\u0CA6", 
    "ಧ": "\u0CA7",
    "ನ": "\u0CA8", 
    "ಪ": "\u0CAA", 
    "ಫ": "\u0CAB", 
    "ಬ": "\u0CAC", 
    "ಭ": "\u0CAD",
    "ಮ": "\u0CAE", 
    "ಯ": "\u0CAF", 
    "ರ": "\u0CB0", 
    "ಲ": "\u0CB2", 
    "ವ": "\u0CB5",
    "ಶ": "\u0CB6", 
    "ಷ": "\u0CB7", 
    "ಸ": "\u0CB8", 
    "ಹ": "\u0CB9", 
    "ಳ": "\u0CB3",
    "mಆ": "\u0CBE", 
    "mಇ": "\u0CBF", 
    "mಈ": "\u0CC0", 
    "mಉ": "\u0CC1", 
    "mಊ": "\u0CC2",
    "mಋ": "\u0CC3", 
    "mಎ": "\u0CC6", 
    "mಏ": "\u0CC7", 
    "mಐ": "\u0CC8", 
    "mಒ": "\u0CCA",
    "mಓ": "\u0CCB", 
    "mಔ": "\u0CCC", 
    "m0.5": "\u0CCD", 
    "ೞ": "\u0CDE",
    "0": "\u0CE6", 
    "೧": "\u0CE7", 
    "೨": "\u0CE8", 
    "೩": "\u0CE9", 
    "೪": "\u0CEA",
    "೫": "\u0CEB", 
    "೬": "\u0CEC", 
    "೭": "\u0CED", 
    "೮": "\u0CEE", 
    "೯": "\u0CEF",
    "mಅ": None,
    "Null": None
}


def Combine(base_character, modifiers=None,ottakshrara = None):
        virma = "\u0CCD"
        result = base_character
        if ottakshrara:
                result += virma + ottakshrara
        if modifiers:
                result += modifiers
        return result




main_directory = input("Enter the path to the main directory: ")
choice = int(input("Select Model: CNN (1) , SVM (2) , KNN (3) :"))
output_lines = []
import os

# Define the base directory for your models
base_model_dir = r"C:\Users\Chaithra M S\Downloads\FYP project\FYP project\Classifier"

if choice == 1:
    loaded_model_base = load_model(os.path.join(base_model_dir, "CNN models", "Base_Character.keras"), compile=False, safe_mode=False)
    loaded_model_mod = load_model(os.path.join(base_model_dir, "CNN models", "Modifier_Character.keras"))
    loaded_model_ott = load_model(os.path.join(base_model_dir, "CNN models", "Ottaksharas_Character.keras"))
elif choice == 2:
    svm_base_model = load(os.path.join(base_model_dir, "SVM models", "svm_base_characters.joblib"))
    svm_mod_model = load(os.path.join(base_model_dir, "SVM models", "svm_mod_characters.joblib"))
    svm_ott_model = load(os.path.join(base_model_dir, "SVM models", "svm_ott_characters.joblib"))
elif choice == 3:
    knn_base_model = load(os.path.join(base_model_dir, "KNN models", "knn_base_hog_characters.joblib"))
    knn_mod_model = load(os.path.join(base_model_dir, "KNN models", "knn_mod_characters.joblib"))
    knn_ott_model = load(os.path.join(base_model_dir, "KNN models", "knn_ott_characters.joblib"))
else:
    print("Error: Improper choice of model")
    exit()

def image_preprocess(image_path):
    image = imread(image_path)
    image_resized = resize(image, (64, 64), anti_aliasing=True)

    if image_resized.ndim == 3 and image_resized.shape[-1] == 4:
        image_resized = image_resized[:, :, :3]

    if image_resized.ndim == 3:
        gray = rgb2gray(image_resized)
    else:
        gray = image_resized  

    features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    features = np.array(features.reshape(1, -1))  

    return features

start_time = time.time()
char_count = 0

for line_num in range(1, 7):
    subdirectory = os.path.join(main_directory, str(line_num))
    if not os.path.exists(subdirectory):
        print(f"Skipping missing directory: {subdirectory}")
        output_lines.append("")
        continue

    line_text = ""

    # Get all matching files and sort numerically
    image_files = sorted(
        [f for f in os.listdir(subdirectory) if f.startswith(f"binary_{line_num}.") and f.endswith((".png", ".jpg", ".jpeg"))],
        key=lambda x: int(x.split(".")[1])  # Extract number after the dot
    )

    for filename in image_files:
        image_path = os.path.join(subdirectory, filename)
        char_count +=1
        if choice == 1:
            img = image.load_img(image_path, target_size=(64, 64))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            pred_base = loaded_model_base.predict(img_array)
            pred_class_base = class_names_base[np.argmax(pred_base)]

            pred_mod = loaded_model_mod.predict(img_array)
            pred_class_mod = class_names_mod[np.argmax(pred_mod)]

            pred_ott = loaded_model_ott.predict(img_array)
            pred_class_ott = class_names_ott[np.argmax(pred_ott)]

            predicted_character = Combine(
                kannada_unicode[pred_class_base],
                kannada_unicode["m" + pred_class_mod],
                kannada_unicode[pred_class_ott]
            )
        
        elif choice == 2:
              feature = image_preprocess(image_path)
              predicted_character = Combine(
                    kannada_unicode[class_names_base[svm_base_model.predict(feature)[0]]],
                    kannada_unicode["m" + class_names_mod[svm_mod_model.predict(feature)[0]]],
                    kannada_unicode[class_names_ott[svm_ott_model.predict(feature)[0]]]
              )
        elif choice == 3:
              feature = image_preprocess(image_path)
              predicted_character = Combine(
                    kannada_unicode[class_names_base[knn_base_model.predict(feature)[0]]],
                    kannada_unicode["m" + class_names_mod[knn_mod_model.predict(feature)[0]]],
                    kannada_unicode[class_names_ott[knn_ott_model.predict(feature)[0]]]
              )

        line_text += predicted_character  # Append prediction to the line

    output_lines.append(line_text)

end_time = time.time()
final_output = "\n".join(output_lines)

print("\nFinal Predicted Text:\n")
print(final_output)
print("")
print(f"Total Recognition time: {end_time - start_time:.4f} seconds")
print(f"Recognition time per character: {(end_time - start_time) / char_count:.4f} seconds")


