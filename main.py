import cv2
from ultralytics import YOLO

model_path = 'models/firev8n.pt' 
video_url = 'video.mp4'

# load the model
model = YOLO(model_path)
# load the video
cap = cv2.VideoCapture(video_url)

while cap.isOpened() :
    # read the video
    ret, frame = cap.read()
    # predict the video frame
    results = model.predict(frame, stream=True)

    for result in results:
        # get the classes names
        classes_names = result.names

        # iterate over each box
        for box in result.boxes:
            # check if confidence is greater than 40 percent
            if box.conf[0] > 0.4:
                # get coordinates
                [x1, y1, x2, y2] = box.xyxy[0]
                # convert to int
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # get the class
                cls = int(box.cls[0])

                # get the class name
                class_name = classes_names[cls]

                # draw the rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

                # put the class name and confidence on the image
                cv2.putText(frame, f'FIRE {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # show the image
    cv2.imshow('window', frame)

    # break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()