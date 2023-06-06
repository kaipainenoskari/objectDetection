# pylint: disable-all
import cv2
import numpy as np
import pyautogui as pg

screenshot = pg.screenshot()

screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

tiles = pg.locateAllOnScreen("wordlePhotos/grayT.png", confidence=0.7)

i = 0
for tile in tiles:
    i += 1
    cv2.rectangle(
        screenshot,
        (tile.left, tile.top),
        (tile.left + tile.width, tile.top + tile.height),
        (0, 255, 0),
        2
    )
    print(tile)

print(i)

cv2.imshow("Screenshot", screenshot)

cv2.waitKey(0)

cv2.destroyAllWindows()