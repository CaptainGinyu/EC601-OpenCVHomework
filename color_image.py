import cv2
import sys

src = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
cv2.namedWindow('Original image', cv2.WINDOW_AUTOSIZE)
cv2.imshow('Original image', src)

blue, green, red = cv2.split(src)
cv2.imshow('Red', red)
cv2.imshow('Green', green)
cv2.imshow('Blue', blue)
cv2.imwrite('red.png', red)
cv2.imwrite('green.png', green)
cv2.imwrite('blue.png', blue)
print('red: ' + str(red[25, 20]))
print('blue: ' + str(blue[25, 20]))
print('green: ' + str(green[25, 20]))

ycrcb_image = cv2.cvtColor(src, cv2.COLOR_BGR2YCR_CB)
y, cb, cr = cv2.split(ycrcb_image)
cv2.imshow('Y', y)
cv2.imshow('Cb', cb)
cv2.imshow('Cr', cr)
cv2.imwrite('y.png', y)
cv2.imwrite('cb.png', cb)
cv2.imwrite('cr.png', cr)
print('y: ' + str(y[25, 20]))
print('cb: ' + str(cb[25, 20]))
print('cr: ' + str(cr[25, 20]))

hsv_image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
hue, saturation, value = cv2.split(hsv_image)
cv2.imshow('Hue', hue)
cv2.imshow('Saturation', saturation)
cv2.imshow('Value', value)
cv2.imwrite('hue.png', hue)
cv2.imwrite('saturation.png', saturation)
cv2.imwrite('value.png', value)
print('hue: ' + str(hue[25, 20]))
print('saturation: ' + str(saturation[25, 20]))
print('value: ' + str(value[25, 20]))

cv2.waitKey(0)
