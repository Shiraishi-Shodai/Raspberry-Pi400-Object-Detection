import cv2

cascade = cv2.CascadeClassifier('../opencv/data/haarcascades/haarcascade_frontalface_alt.xml')
# 顔を検出するための画像を読込む
img = cv2.imread('img/mona_lisa.jpg')

# 顔を検出する
# 戻り値としては検出した画像の座標情報が返却される
facerect = cascade.detectMultiScale(img)

#検出した顔を四角い枠線で囲む (検出した顔の数だけ for で繰り返す)
for rect in facerect:
    cv2.rectangle(img, tuple(rect[0:2]),tuple(rect[0:2] + rect[2:4]), (255, 255, 255), thickness=2)

# 画像を表示する
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

