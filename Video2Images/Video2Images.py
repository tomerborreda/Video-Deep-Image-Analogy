import cv2

# Opens the Video file
cap = cv2.VideoCapture('input_video.mp4')
i = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    # Make a 720p photo a square
    begin = frame.shape[1]//2 - 720//2 #begin = frame.shape[1]//2 - 1080//2
    end = frame.shape[1]//2 + 720//2 #end = frame.shape[1]//2 + 1080//2
    frame = frame[:, begin:end, :]
    # Resize the photo
    frame = cv2.resize(frame,(448,448))
    # Save each frame
    cv2.imwrite('output_frames' + '/' + str(i) + '.jpg', frame)
    i += 1

cap.release()
cv2.destroyAllWindows()