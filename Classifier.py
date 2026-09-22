from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

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


loaded_model_base = load_model("Base_Character.keras")
loaded_model_mod = load_model("Modifier_Character.keras")
loaded_model_ott = load_model("Ottaksharas_Character.keras")

test_image_path =input("Enter path of image: ")


# Load and preprocess the image
img = image.load_img(test_image_path, target_size=(64, 64))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)


# Display the image
plt.imshow(img)
plt.grid(False)


# Make a prediction
pred_base = loaded_model_base.predict(img_array)
pred_class_base = class_names_base[np.argmax(pred_base)]

pred_mod = loaded_model_mod.predict(img_array)
pred_class_mod = class_names_mod[np.argmax(pred_mod)]

pred_ott = loaded_model_ott.predict(img_array)
pred_class_ott = class_names_ott[np.argmax(pred_ott)]


print("Character predicited: ", Combine(kannada_unicode[pred_class_base],kannada_unicode["m"+pred_class_mod],kannada_unicode[pred_class_ott]))