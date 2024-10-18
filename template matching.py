import cv2
import os
import pyautogui
import numpy as np
from time import sleep
from pynput.keyboard import Controller

keyboard = Controller()


# Load digit templates
def load_digit_templates():
    """Loads the digit templates (0-9) from the 'templates' folder."""
    templates = {}
    for digit in range(10):
        template_path = f'templates/{digit}.png'
        template = cv2.imread(template_path, 0)  # Load as grayscale

        if template is None:
            print(f"Error: Template {template_path} could not be loaded. Check the file path or integrity.")
            raise FileNotFoundError(f"Template {digit}.png not found or corrupted in 'templates' folder.")

        templates[digit] = template

    return templates


# Perform template matching on a captured screenshot
def match_template(screenshot_path, templates):
    """Matches digit templates with the screenshot."""
    screenshot = cv2.imread(screenshot_path, 0)  # Load screenshot in grayscale
    print(f"Screenshot size: {screenshot.shape}")  # Log screenshot size
    detected_digits = []

    # Loop through each digit template (0-9)
    for digit, template in templates.items():
        print(f"Template {digit} size: {template.shape}")  # Log template size

        if template.shape[0] > screenshot.shape[0] or template.shape[1] > screenshot.shape[1]:
            print(f"Skipping template {digit} as it is larger than the screenshot.")
            continue

        result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
        print(f"Match result for digit {digit}: {np.max(result)}")  # Log matching result
        threshold = 0.6  # Lower threshold to 0.6 for more lenient matching
        locations = np.where(result >= threshold)

        # If matches found, append the detected digit
        for loc in zip(*locations[::-1]):
            print(f"Detected digit {digit} at {loc}")
            detected_digits.append((digit, loc))
            cv2.rectangle(screenshot, loc, (loc[0] + template.shape[1], loc[1] + template.shape[0]), (0, 255, 0), 2)

    # Save the screenshot with matching regions marked for debugging
    cv2.imwrite('picture/matched_image.png', screenshot)

    # Sort the digits by their x-coordinate (assuming left-to-right order)
    detected_digits.sort(key=lambda x: x[1][0])

    # Return the detected digits as a number
    detected_number = ''.join(str(digit) for digit, _ in detected_digits)
    return detected_number if detected_number else None


def capture_game_region(region):
    """Capture the game region where digits are displayed."""
    screenshot_path = 'picture/captured_image.png'
    im = pyautogui.screenshot(region=region)
    im.save(screenshot_path)
    print(f"Screenshot saved at {screenshot_path}")
    return screenshot_path


def Miner():
    # Define the region where the number is displayed (adjust the region as needed)
    game_region = (1075, 900, 1215 - 1075, 1000 - 900)
    templates = load_digit_templates()  # Load digit templates

    while True:
        # Capture the game region
        screenshot_path = capture_game_region(game_region)

        # Perform template matching to detect digits
        detected_number = match_template(screenshot_path, templates)

        if detected_number:
            print(f"Detected number: {detected_number}")
            pyautogui.press('f6')  # Bring up the input field
            keyboard.type(f"/dig {detected_number}\n")
            sleep(2)  # Adjust sleep before retrying
        else:
            print("No digits detected, retrying...")

        # Delay between attempts
        sleep(5)  # You can adjust this delay as needed


# Start the miner
Miner()
