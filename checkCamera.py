from cv2 import *

img = cv2.imread("images/test1/test/img_11.jpg",1)

resized = cv2.resize(img, (20,20))

cv2.imshow("images/test1/test/img_11.jpg", resized)

cv2.waitKey(0)

cv2.destroyAllWindows()