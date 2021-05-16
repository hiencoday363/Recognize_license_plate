import cv2

from lib_detection import load_model, detect_lp, im2single, preprocess, licensePlate

# Đường dẫn ảnh
img_path = "test/test08.jpg"

# Load model LP detection
wpod_net_path = "wpod-net_update1.json"
wpod_net = load_model(wpod_net_path)

# Đọc file ảnh đầu vào
Ivehicle = cv2.imread(img_path)
# cv2.imshow("Anh goc", Ivehicle)

# Kích thước lớn nhất và nhỏ nhất của 1 chiều ảnh
Dmax = 608
Dmin = 288

# Lấy tỷ lệ giữa W và H của ảnh và tìm ra chiều nhỏ nhất
ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
side = int(ratio * Dmin)
bound_dim = min(side, Dmax)

Nhan, LpImg, lp_type, listPoint = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.5)

if len(LpImg) and len(listPoint):
    for id, tuplePoint in enumerate(listPoint):
        cv2.imshow("LpImg", LpImg[id])

        # draw rectangle on img
        cv2.rectangle(Ivehicle, tuplePoint[0], tuplePoint[1], (255, 0, 0), 2)
        img_process = preprocess(LpImg[id])
        text = licensePlate(img_process)

        cv2.rectangle(Ivehicle, tuplePoint[0], (tuplePoint[1][0],tuplePoint[0][1]-20), (255, 255, 255), -1)

        # write license on img
        cv2.putText(Ivehicle, text, tuplePoint[0], cv2.FONT_HERSHEY_PLAIN, 1.5 , (0, 0, 255), lineType=cv2.LINE_AA)

cv2.imshow("Bien so", Ivehicle)
cv2.waitKey(0)

cv2.destroyAllWindows()
