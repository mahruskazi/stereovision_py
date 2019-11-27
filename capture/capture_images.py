import cv2
import time
import re
import os

class CaptureImages:

    left_camera_id = 0
    right_camera_id = 1
    height = 480
    width = 640

    left_cap = None
    right_cap = None

    def setup_capture(self):
        f = open("capture/camera_settings.txt", "r")

        req = re.search('left_camera = (\d+), right_camera = (\d+), height = (\d+), width = (\d+)', f.read())
        if req:
            self.left_camera_id = int(req.group(1))
            self.right_camera_id = int(req.group(2))
            self.height = int(req.group(3))
            self.width = int(req.group(4))
        else:
            print('Could not parse camera_settings.txt, please check if settings are in the right format')

        self.left_cap = cv2.VideoCapture(self.left_camera_id)
        self.right_cap = cv2.VideoCapture(self.right_camera_id)

        self.left_cap.set(3, self.width)
        self.left_cap.set(4, self.height)

        self.right_cap.set(3, self.width)
        self.right_cap.set(4, self.height)

    def save_images(self, interval = 5):

        image = 0
        t = time.process_time()
        print("Starting image capture with interval: " + str(interval))
        print("Press 'q' to stop")

        while(True):
            # Capture frame-by-frame
            _, left_frame = self.left_cap.read()
            _, right_frame = self.right_cap.read()

            # Display the resulting frame
            cv2.imshow('left frame',left_frame)
            cv2.imshow('right frame',right_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if ((time.process_time() - t) > interval):
                cv2.imwrite('capture/pairs/left_'+str(image).zfill(2)+'.png',left_frame)
                cv2.imwrite('capture/pairs/right_'+str(image).zfill(2)+'.png',right_frame)
                print("Write: " + str(image))
                image += 1
                t = time.process_time()

        cv2.destroyAllWindows()

    def capture_scene(self):
        print("Press 's' to save the scene or 'q' to stop")

        while(True):
            # Capture frame-by-frame
            _, left_frame = self.left_cap.read()
            _, right_frame = self.right_cap.read()

            # Display the resulting frame
            cv2.imshow('left frame',left_frame)
            cv2.imshow('right frame',right_frame)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('s'):
                cv2.imwrite('tuner/scene/left_scene.png',left_frame)
                cv2.imwrite('tuner/scene/right_scene.png',right_frame)
                print('Saving to tuner/scene/')

        cv2.destroyAllWindows()

    def release_cameras(self):
        self.left_cap.release()
        self.right_cap.release()

