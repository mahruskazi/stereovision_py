import cv2
import numpy as np
import time
import cv2
import numpy as np
import json
from stereovision.calibration import StereoCalibrator
from stereovision.calibration import StereoCalibration
from datetime import datetime

class Tracking:

    # Depth map default preset
    SWS = 5
    PFS = 5
    PFC = 29
    MDS = -30
    NOD = 160
    TTH = 100
    UR = 10
    SR = 14
    SPWS = 100

    img = None
    orig_img = None

    drawing = False # true if mouse is pressed
    prev_drawing = False
    ix,iy = -1,-1
    x,y = -1,-1

    px, py = -1,-1

    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.15

    img_object = None

    def __init__(self, left_cap, right_cap):
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.draw_rectangle)

        # Initialize interface windows
        cv2.namedWindow("Depth Map")
        cv2.moveWindow("Depth Map", 50,100)
        cv2.setMouseCallback('Depth Map', self.click_location)

        calibration = StereoCalibration(input_folder='calibrate/calib_result')

        sbm = cv2.StereoBM_create(numDisparities=0, blockSize=21)

        self.load_map_settings("tuner/map_settings.txt", sbm)
        orb = cv2.ORB_create(self.MAX_FEATURES)
        while(True):
            # Capture frame-by-frame
            _, left_frame = left_cap.read()
            _, right_frame = right_cap.read()

            imgLeft = cv2.cvtColor (left_frame, cv2.COLOR_BGR2GRAY)
            imgRight = cv2.cvtColor (right_frame, cv2.COLOR_BGR2GRAY)
            rectified_pair = calibration.rectify((imgLeft, imgRight))
            if not self.drawing:
                self.img = rectified_pair[0]
            cv2.imshow("image", self.img)
            depth_map = self.stereo_depth_map(rectified_pair, sbm)

            distance = self.get_distance(depth_map[0])
            cv2.putText(depth_map[1], str(distance), (10,450), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Depth Map", depth_map[1])

            if(self.prev_drawing):
                self.img_object = np.copy(rectified_pair[0])[self.iy:self.y, self.ix:self.x]
                self.prev_drawing = False

            if(self.img_object is not None and self.img_object.size > 0):
                cv2.imshow('cropped', self.img_object)
                self.track_object(orb, self.img_object, rectified_pair[0])

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def stereo_depth_map(self, rectified_pair, sbm):
        dmLeft = rectified_pair[0]
        dmRight = rectified_pair[1]
        disparity = sbm.compute(dmLeft, dmRight)
        local_max = disparity.max()
        local_min = disparity.min()
        disparity_grayscale = (disparity-local_min)*(65535.0/(local_max-local_min))
        disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0/65535.0))
        disparity_color = cv2.applyColorMap(disparity_fixtype, cv2.COLORMAP_JET)
        return [disparity_grayscale, disparity_color]

    def load_map_settings(self, fName, sbm):
        print('Loading parameters from file...')
        f=open(fName, 'r')
        data = json.load(f)
        self.SWS=data['SADWindowSize']
        self.PFS=data['preFilterSize']
        self.PFC=data['preFilterCap']
        self.MDS=data['minDisparity']
        self.NOD=data['numberOfDisparities']
        self.TTH=data['textureThreshold']
        self.UR=data['uniquenessRatio']
        self.SR=data['speckleRange']
        self.SPWS=data['speckleWindowSize']
        #sbm.setSADWindowSize(SWS)
        sbm.setPreFilterType(1)
        sbm.setPreFilterSize(self.PFS)
        sbm.setPreFilterCap(self.PFC)
        sbm.setMinDisparity(self.MDS)
        sbm.setNumDisparities(self.NOD)
        sbm.setTextureThreshold(self.TTH)
        sbm.setUniquenessRatio(self.UR)
        sbm.setSpeckleRange(self.SR)
        sbm.setSpeckleWindowSize(self.SPWS)
        f.close()
        print ('Parameters loaded from file '+fName)

    def get_distance(self, depth_map):
        # Slope and y_shift is created by plotting values for objects at different
        # distances and finding the line of best fit
        # print(depth_map[self.py][self.px]) # Uncomment to help tune values
        slope = -0.00185
        y_shift = 152.7238
        return slope*depth_map[self.py][self.px] + y_shift

    def track_object(self, orb, img_object, img_scene):
        keypoints_obj, descriptors_obj = orb.detectAndCompute(img_object, None)
        keypoints_scene, descriptors_scene = orb.detectAndCompute(img_scene, None)

        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors_obj, descriptors_scene, None)

        matches.sort(key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        numGoodMatches = int(len(matches) * self.GOOD_MATCH_PERCENT)
        good_matches = matches[:numGoodMatches]

        # cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)

        #-- Draw matches
        imMatches = cv2.drawMatches(img_object, keypoints_obj, img_scene, keypoints_scene, good_matches, None)

        #-- Localize the object
        obj = np.empty((len(good_matches),2), dtype=np.float32)
        scene = np.empty((len(good_matches),2), dtype=np.float32)
        for i in range(len(good_matches)):
            #-- Get the keypoints from the good matches
            obj[i,0] = keypoints_obj[good_matches[i].queryIdx].pt[0]
            obj[i,1] = keypoints_obj[good_matches[i].queryIdx].pt[1]
            scene[i,0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
            scene[i,1] = keypoints_scene[good_matches[i].trainIdx].pt[1]
        H = None
        if(obj.size > 0 and scene.size > 0):
            H, _ =  cv2.findHomography(obj, scene, cv2.RANSAC)
        if (H is not None):
            #-- Get the corners from the image_1 ( the object to be "detected" )
            obj_corners = np.empty((4,1,2), dtype=np.float32)
            obj_corners[0,0,0] = 0
            obj_corners[0,0,1] = 0
            obj_corners[1,0,0] = img_object.shape[1]
            obj_corners[1,0,1] = 0
            obj_corners[2,0,0] = img_object.shape[1]
            obj_corners[2,0,1] = img_object.shape[0]
            obj_corners[3,0,0] = 0
            obj_corners[3,0,1] = img_object.shape[0]
            scene_corners = cv2.perspectiveTransform(obj_corners, H)

            #-- Draw lines between the corners (the mapped object in the scene - image_2 )
            cv2.line(imMatches, (int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])),\
                (int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])), (0,255,0), 4)
            cv2.line(imMatches, (int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])),\
                (int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])), (0,255,0), 4)
            cv2.line(imMatches, (int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])),\
                (int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])), (0,255,0), 4)
            cv2.line(imMatches, (int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])),\
                (int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])), (0,255,0), 4)
            #-- Show detected matches
            cv2.imshow('Good Matches & Object detection', imMatches)


    def draw_rectangle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix,self.iy = x,y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                cv2.rectangle(self.img,(int(self.ix),int(self.iy)),(x,y),(0,255,0))
        elif event == cv2.EVENT_LBUTTONUP:
            self.prev_drawing = True
            self.drawing = False
            self.x = x
            self.y = y
            cv2.rectangle(self.img,(int(self.ix),int(self.iy)),(x,y),(0,255,0))

    def click_location(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.px = x
            self.py = y