import cv2
import os
import pytesseract
import numpy as np
import pyautogui
from PIL import Image, ImageEnhance
from pynput.keyboard import Controller
from time import sleep
import random
import re
import json

keyboard = Controller()

# Statistics for success and failure counts
success_count = 0
fail_count = 0
# Set to store previously attempted numbers
attempted_numbers = set()
# Correction mapping for incorrect -> correct numbers
correction_mapping = {}


def capture_console_output(region):
    """Capture the console output for OCR or image matching."""
    sleep(2)  # Delay to ensure the console has updated
    print(f"Capturing console output from region: {region}")  # Log region info

    # Take a screenshot of the console/chat region
    try:
        im = pyautogui.screenshot(region=region)  # Capture the region
        screenshot_path = '../mineworker/picture/console_output.png'

        # Ensure directory exists
        if not os.path.exists('../mineworker/picture'):
            os.makedirs('../mineworker/picture')

        im.save(screenshot_path)
        print(f"Console output saved at {screenshot_path}")  # Log where the image is saved
        return screenshot_path
    except Exception as e:
        print(f"Failed to capture console output: {e}")
        return None


def match_console_image(region=(900, 975, 120, 50), template_path='picture/console_example.png', threshold=0.8):
    """Match the captured console output with the example template image."""
    console_screenshot_path = capture_console_output(region)
    if not console_screenshot_path:
        print("Console screenshot was not captured. Skipping image matching.")
        return False

    # Load the captured screenshot and the template image
    console_img = cv2.imread(console_screenshot_path, 0)  # Load the captured image in grayscale
    template_img = cv2.imread(template_path, 0)  # Load the template image in grayscale

    # Verify that the template image exists
    if template_img is None:
        print(f"Template image '{template_path}' not found!")
        return False

    # Perform template matching
    result = cv2.matchTemplate(console_img, template_img, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    print(f"Image match score: {max_val}")  # Print the matching score
    if max_val >= threshold:
        print(f"Image match successful with a score of {max_val}.")
        return True
    else:
        print(f"Image match failed. Best match score: {max_val}.")
        return False


# Function to update and log success rate
def log_success_rate():
    total_attempts = success_count + fail_count
    success_rate = (success_count / total_attempts) * 100 if total_attempts > 0 else 0
    print(f"Success count: {success_count}, Fail count: {fail_count}, Success rate: {success_rate:.2f}%")


# Load correction mapping from file
def load_correction_mapping():
    global correction_mapping
    if os.path.exists('../dig 1009/correction_mapping.json'):
        with open('../dig 1009/correction_mapping.json', 'r') as file:
            correction_mapping = json.load(file)
    else:
        correction_mapping = {}


# Function to check if a number has already been attempted
def is_repeated_attempt(detected_digits):
    if detected_digits in attempted_numbers:
        print(f"Number {detected_digits} has already been attempted. Skipping to avoid repeated attempts.")
        return True
    return False


# Save correction mapping to file
def save_correction_mapping():
    with open('../dig 1009/correction_mapping.json', 'w') as file:
        json.dump(correction_mapping, file)


# Load attempted numbers from a file
def load_attempted_numbers():
    global attempted_numbers
    if os.path.exists('attempted_numbers.json'):
        with open('attempted_numbers.json', 'r') as file:
            attempted_numbers = set(json.load(file))
    else:
        attempted_numbers = set()


# Save attempted numbers to a file
def save_attempted_numbers():
    with open('attempted_numbers.json', 'w') as file:
        json.dump(list(attempted_numbers), file)


# Correct detected number based on previous corrections
def correct_detected_number(detected_digits):
    if detected_digits in correction_mapping:
        corrected_digits = correction_mapping[detected_digits]
        print(f"Applying correction: {detected_digits} -> {corrected_digits}")
        return corrected_digits
    return detected_digits


# Start of the Miner function
def Miner():
    global success_count, fail_count, attempted_numbers, correction_mapping

    load_attempted_numbers()  # Load previous attempts
    load_correction_mapping()  # Load previous correction mappings

    while True:
        sleep(5)  # Time to switch to the window if needed

        # File paths
        directory = 'picture'
        screenshot_path = os.path.join(directory, 'captured_image.png')
        processed_path = os.path.join(directory, 'processed_image.png')
        thresholded_path = os.path.join(directory, 'thresholded_image.png')
        high_dpi_path = os.path.join(directory, 'high_dpi_image.png')

        if not os.path.exists(directory):
            os.makedirs(directory)

        # Take a screenshot (adjust the region based on your screen)
        im = pyautogui.screenshot(region=(1075, 900, 1215 - 1075, 1000 - 900))
        im.save(screenshot_path)
        print(f"Screenshot saved at {screenshot_path}")

        # Read the screenshot and process it
        img = cv2.imread(screenshot_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Upscale image for better OCR
        scale_percent = 400  # Increased upscale percentage
        width = int(gray.shape[1] * scale_percent / 100)
        height = int(gray.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_gray = cv2.resize(gray, dim,
                                  interpolation=cv2.INTER_CUBIC)  # Use INTER_CUBIC for better quality upscaling

        # Increase contrast and brightness
        pil_img = Image.fromarray(resized_gray)
        enhancer = ImageEnhance.Contrast(pil_img)
        contrast_img = enhancer.enhance(4.0)  # Increase contrast even more for better readability
        contrast_img = np.array(contrast_img)

        # Apply median filtering to remove salt-and-pepper noise
        filtered_img = cv2.medianBlur(contrast_img, 3)

        # Apply Gaussian Blur to reduce noise
        blurred_img = cv2.GaussianBlur(filtered_img, (3, 3), 0)

        # Apply sharpening with a more aggressive kernel
        kernel = np.array([[0, -1, 0], [-1, 9, -1], [0, -1, 0]])
        sharpened_img = cv2.filter2D(blurred_img, -1, kernel)

        # Noise Reduction with a stronger filter
        denoised_img = cv2.fastNlMeansDenoising(sharpened_img, None, 60, 7, 21)

        # Morphological Transformations (Erosion/Dilation)
        kernel = np.ones((3, 3), np.uint8)  # Slightly larger kernel
        morph_img = cv2.morphologyEx(denoised_img, cv2.MORPH_CLOSE, kernel)  # Using closing to reduce noise

        # Apply edge detection
        edges = cv2.Canny(morph_img, 100, 200)

        # Save images for comparison
        cv2.imwrite(high_dpi_path, morph_img)
        print(f"High-DPI image saved at {high_dpi_path}")

        # Apply adaptive thresholding with edge detection
        adaptive_thresh = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)
        cv2.imwrite(thresholded_path, adaptive_thresh)
        print(f"Thresholded image saved at {thresholded_path}")

        # OCR using Tesseract
        pytesseract.pytesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

        detected_digits = None
        success = False  # Track whether the detection was successful
        replacement_attempted = False  # Track if we have already replaced '3' with '9'

        # Run OCR on each image variation with multiple psm modes
        images = [high_dpi_path, thresholded_path, screenshot_path, processed_path]
        for img_path in images:
            print(f"\nTrying OCR on {img_path}")
            for psm in [3, 6, 7, 8, 10, 13]:
                ocr_output = pytesseract.image_to_string(img_path,
                                                         config=f'--psm {psm} --oem 3 -c tessedit_char_whitelist=0123456789')
                print(f"OCR raw output with --psm {psm}: {ocr_output.strip()}")

                # Extract valid digits using regular expressions
                valid_digits = re.findall(r'\b\d{3,4}\b', ocr_output)
                if valid_digits:
                    detected_digits = valid_digits[0]
                    detected_digits = correct_detected_number(detected_digits)  # Apply corrections if any

                    # Check confidence score for detected digits
                    data = pytesseract.image_to_data(img_path, output_type=pytesseract.Output.DICT)
                    for i, word in enumerate(data['text']):
                        if word == detected_digits and int(data['conf'][i]) > 85:
                            print(f"Detected digits with high confidence: {detected_digits}")
                            success = True
                            break

                    # Stop trying other PSMs once digits are detected and valid
                    if success:
                        break
            if success:
                break  # Stop checking other images if digits are detected

        # If valid digits are detected, type it
        if detected_digits:
            print(f"Typing detected number: {detected_digits}")
            pyautogui.press('f6')  # Bring up the input field
            keyboard.type(f"/dig {detected_digits}\n")

            # Verification step: check if the detected number was entered correctly via image matching
            if match_console_image():
                print(f"Command successful with detected number: {detected_digits}")
                success = True
                # Add the successfully attempted number to the set
                attempted_numbers.add(detected_digits)
                save_attempted_numbers()  # Save immediately
            else:
                print(f"Command verification failed for: {detected_digits}")
                success = False

                # If verification failed and the detected number contains a '3', attempt replacing '3' with '9'
                if '3' in detected_digits and not replacement_attempted:
                    print(f"Verification failed and detected digits contain '3'. Replacing '3' with '9' and retrying.")
                    corrected_digits = detected_digits.replace('3', '9')
                    correction_mapping[detected_digits] = corrected_digits  # Store the correction

                    # Check if the number has been attempted before
                    if is_repeated_attempt(detected_digits):
                        detected_digits = None
                        continue  # Skip to the next attempt

                    save_correction_mapping()  # Save immediately
                    detected_digits = corrected_digits
                    replacement_attempted = True

                    # Retype the number with '3' replaced by '9'
                    pyautogui.press('f6')  # Bring up the input field again
                    keyboard.type(f"/dig {detected_digits}\n")

                    # Retry verification
                    if match_console_image():
                        print(f"Command successful after replacing '3' with '9': {detected_digits}")
                        success = True
                        # Add the successfully attempted number to the set
                        attempted_numbers.add(detected_digits)
                        save_attempted_numbers()  # Save immediately
                    else:
                        print(f"Command verification failed even after replacing '3' with '9'.")
                        success = False

        else:
            print("No valid digits detected after all attempts.")

            # Update success and failure counts
            if success:
                success_count += 1
                log_success_rate()
                # Reset attempted numbers set after a success
                attempted_numbers = set()
            else:
                fail_count += 1
                log_success_rate()

            save_attempted_numbers()  # Save attempted numbers after each try
            save_correction_mapping()  # Save correction mapping after each try

            # Set different delays based on success or failure
            if success:
                delay = random.randint(40, 50)  # Longer delay on success
                print(f"Waiting for {delay} seconds before the next attempt...")
            else:
                delay = random.randint(1, 4)  # Shorter delay on failed verification
                print(f"Waiting for {delay} seconds (shorter) before retrying...")

            sleep(delay)


Miner()
