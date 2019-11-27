import os
import cv2
from stereovision.calibration import StereoCalibrator
from stereovision.calibration import StereoCalibration
from stereovision.exceptions import ChessboardNotFoundError

class CalibrateCameras:

    # Global
    total_photos = 47
    img_width = 640
    img_height = 480

    # Chessboard parameters
    rows = 6
    columns = 9
    square_size = 2.5

    def start_calibration(self):
        calibrator = StereoCalibrator(self.rows, self.columns, self.square_size, (self.img_width, self.img_height))
        photo_counter = 0
        print ("Start calibration, press any key on image to move to the next")

        while photo_counter != self.total_photos:
            print('Import pair No ' + str(photo_counter))
            leftName = 'capture/pairs/left_'+str(photo_counter).zfill(2)+'.png'
            rightName = 'capture/pairs/right_'+str(photo_counter).zfill(2)+'.png'

            photo_counter = photo_counter + 1
            if os.path.isfile(leftName) and os.path.isfile(rightName):
                imgLeft = cv2.imread(leftName,1)
                imgRight = cv2.imread(rightName,1)
                try:
                    calibrator._get_corners(imgLeft)
                    calibrator._get_corners(imgRight)
                except ChessboardNotFoundError as error:
                    print (error)
                    print ("Pair No "+ str(photo_counter) + " ignored")
                else:
                    calibrator.add_corners((imgLeft, imgRight), True)

        print ('End cycle')


        print ('Starting calibration... It can take several minutes!')
        calibration = calibrator.calibrate_cameras()
        calibration.export('calibrate/calib_result')
        print ('Calibration complete!')


        # Lets rectify and show last pair after  calibration
        calibration = StereoCalibration(input_folder='calibrate/calib_result')
        rectified_pair = calibration.rectify((imgLeft, imgRight))

        cv2.imshow('Left CALIBRATED', rectified_pair[0])
        cv2.imshow('Right CALIBRATED', rectified_pair[1])
        cv2.imwrite("calibrate/rectifyed_left.jpg",rectified_pair[0])
        cv2.imwrite("calibrate/rectifyed_right.jpg",rectified_pair[1])
        cv2.waitKey(0)

