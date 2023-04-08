import cv2 as cv

img = cv.imread("test.jpeg")
# cv.imshow("img", img)
# cv.waitKey(0)
# print(img)
# print(type(img))
# print(len(img[0][0]), len(img[0]), len(img))


gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow("gray", gray)
# cv.waitKey(0)
print(gray)
print(type(gray))
print(len(gray[0]), len(gray))
# [[[ 64  29  19]
#   [ 63  19   2]
#   [ 91  27   2]
#   ...
#   [ 39 177 176]
#   [ 66 208 203]
#   [ 52 197 188]]
#
#  [[ 67  31  25]
#   [ 63  18   7]
#   [ 86  21   0]
#   ...
#   [ 60 195 197]
#   [ 63 202 198]
#   [ 37 179 172]]
#
# ...
#
#  [[ 85  45  50]
#   [ 67  19  17]
#   [ 81  18   4]
#   ...
#   [102 230 235]
#   [ 85 219 219]
#   [ 56 193 189]]]

# [[ 30  19  27 ... 161 190 178]
#  [ 33  20  22 ... 180 185 161]
#  [ 51  24  21 ... 217 204 176]
#  ...
#  [ 26  25  23 ...  22  20  16]
#  [ 28  26  24 ...  23  22  19]
#  [ 28  27  24 ...  24  23  20]]