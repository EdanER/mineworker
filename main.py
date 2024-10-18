import pyautogui
from time import sleep
sleep(3)
screenshot = pyautogui.screenshot(region=(1075, 900, 140, 100))  # Adjust region based on where digits are
screenshot.save("game_digits.png")
