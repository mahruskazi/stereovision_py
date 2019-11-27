import cv2
import os
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
import json
from stereovision.calibration import StereoCalibrator
from stereovision.calibration import StereoCalibration


class Tuner:

    # Depth map function
    SWS = 5
    PFS = 5
    PFC = 29
    MDS = -25
    NOD = 128
    TTH = 100
    UR = 10
    SR = 15
    SPWS = 100

    loading_settings = 0
    dmObject = None
    rectified_pair = []
    axcolor = 'lightgoldenrodyellow'
    SWSaxe = None
    PFSaxe = None
    PFCaxe = None
    MDSaxe = None
    NODaxe = None
    TTHaxe = None
    URaxe = None
    SRaxe = None
    SPWSaxe = None

    sSWS = None
    sPFS = None
    sPFC = None
    sMDS = None
    sNOD = None
    sTTH = None
    sUR = None
    sSR = None
    sSPWS = None

    buttons = None
    buttonl = None

    def __init__(self):
        imgLeft = cv2.imread('tuner/scene/left_scene.png',0) # pair_img [0:photo_height,0:image_width] #Y+H and X+W
        imgRight = cv2.imread('tuner/scene/right_scene.png',0)# pair_img [0:photo_height,image_width:photo_width] #Y+H and X+W


        # Implementing calibration data
        print('Read calibration data and rectifying stereo pair...')
        calibration = StereoCalibration(input_folder='calibrate/calib_result')
        self.rectified_pair = calibration.rectify((imgLeft, imgRight))

        disparity = self.stereo_depth_map(self.rectified_pair)

        # Set up and draw interface
        # Draw left image and depth map
        plt.subplots_adjust(left=0.15, bottom=0.5)
        plt.subplot(1,2,1)
        self.dmObject = plt.imshow(self.rectified_pair[0], 'gray')

        saveax = plt.axes([0.3, 0.38, 0.15, 0.04]) #stepX stepY width height
        self.buttons = Button(saveax, 'Save settings', color=self.axcolor, hovercolor='0.975')

        self.buttons.on_clicked(self.save_map_settings)

        loadax = plt.axes([0.5, 0.38, 0.15, 0.04]) #stepX stepY width height
        self.buttonl = Button(loadax, 'Load settings', color=self.axcolor, hovercolor='0.975')
        self.buttonl.on_clicked(self.load_map_settings)

        plt.subplot(1,2,2)
        self.dmObject = plt.imshow(disparity, aspect='equal', cmap='jet')

        self.SWSaxe = plt.axes([0.15, 0.01, 0.7, 0.025], facecolor=self.axcolor) #stepX stepY width height
        self.PFSaxe = plt.axes([0.15, 0.05, 0.7, 0.025], facecolor=self.axcolor) #stepX stepY width height
        self.PFCaxe = plt.axes([0.15, 0.09, 0.7, 0.025], facecolor=self.axcolor) #stepX stepY width height
        self.MDSaxe = plt.axes([0.15, 0.13, 0.7, 0.025], facecolor=self.axcolor) #stepX stepY width height
        self.NODaxe = plt.axes([0.15, 0.17, 0.7, 0.025], facecolor=self.axcolor) #stepX stepY width height
        self.TTHaxe = plt.axes([0.15, 0.21, 0.7, 0.025], facecolor=self.axcolor) #stepX stepY width height
        self.URaxe = plt.axes([0.15, 0.25, 0.7, 0.025], facecolor=self.axcolor) #stepX stepY width height
        self.SRaxe = plt.axes([0.15, 0.29, 0.7, 0.025], facecolor=self.axcolor) #stepX stepY width height
        self.SPWSaxe = plt.axes([0.15, 0.33, 0.7, 0.025], facecolor=self.axcolor) #stepX stepY width height

        self.sSWS = Slider(self.SWSaxe, 'SWS', 5.0, 255.0, valinit=5)
        self.sPFS = Slider(self.PFSaxe, 'PFS', 5.0, 255.0, valinit=5)
        self.sPFC = Slider(self.PFCaxe, 'PreFiltCap', 5.0, 63.0, valinit=29)
        self.sMDS = Slider(self.MDSaxe, 'MinDISP', -100.0, 100.0, valinit=-25)
        self.sNOD = Slider(self.NODaxe, 'NumOfDisp', 16.0, 256.0, valinit=128)
        self.sTTH = Slider(self.TTHaxe, 'TxtrThrshld', 0.0, 1000.0, valinit=100)
        self.sUR = Slider(self.URaxe, 'UnicRatio', 1.0, 20.0, valinit=10)
        self.sSR = Slider(self.SRaxe, 'SpcklRng', 0.0, 40.0, valinit=15)
        self.sSPWS = Slider(self.SPWSaxe, 'SpklWinSze', 0.0, 300.0, valinit=100)

        self.sSWS.on_changed(self.update)
        self.sPFS.on_changed(self.update)
        self.sPFC.on_changed(self.update)
        self.sMDS.on_changed(self.update)
        self.sNOD.on_changed(self.update)
        self.sTTH.on_changed(self.update)
        self.sUR.on_changed(self.update)
        self.sSR.on_changed(self.update)
        self.sSPWS.on_changed(self.update)

        print('Show interface to user')
        plt.show()

    def stereo_depth_map(self, rectified_pair):
        print ('SWS='+str(self.SWS)+' PFS='+str(self.PFS)+' PFC='+str(self.PFC)+' MDS='+\
            str(self.MDS)+' NOD='+str(self.NOD)+' TTH='+str(self.TTH))
        print (' UR='+str(self.UR)+' SR='+str(self.SR)+' SPWS='+str(self.SPWS))
        c, r = rectified_pair[0].shape
        disparity = np.zeros((c, r), np.uint8)
        sbm = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        sbm.setPreFilterType(1)
        sbm.setPreFilterSize(self.PFS)
        sbm.setPreFilterCap(self.PFC)
        sbm.setMinDisparity(self.MDS)
        sbm.setNumDisparities(self.NOD)
        sbm.setTextureThreshold(self.TTH)
        sbm.setUniquenessRatio(self.UR)
        sbm.setSpeckleRange(self.SR)
        sbm.setSpeckleWindowSize(self.SPWS)
        dmLeft = rectified_pair[0]
        dmRight = rectified_pair[1]
        disparity = sbm.compute(dmLeft, dmRight)
        local_max = disparity.max()
        local_min = disparity.min()
        disparity_visual = (disparity-local_min)*(1.0/(local_max-local_min))
        local_max = disparity_visual.max()
        local_min = disparity_visual.min()
        return disparity_visual

    def save_map_settings(self, event):
        self.buttons.label.set_text ("Saving...")
        print('Saving to file...')
        result = json.dumps({'SADWindowSize': self.SWS, 'preFilterSize': self.PFS, 'preFilterCap': self.PFC, \
                'minDisparity': self.MDS, 'numberOfDisparities': self.NOD, 'textureThreshold': self.TTH, \
                'uniquenessRatio': self.UR, 'speckleRange': self.SR, 'speckleWindowSize': self.SPWS},\
                sort_keys=True, indent=4, separators=(',',':'))
        fName = 'tuner/map_settings.txt'
        f = open(str(fName), 'w')
        f.write(result)
        f.close()
        self.buttons.label.set_text ("Save to file")
        print ('Settings saved to file '+fName)

    #Update depth map parameters and redraw
    def update(self, val):
        loading_settings = 0
        self.SWS = int(self.sSWS.val/2)*2+1 #convert to ODD
        self.PFS = int(self.sPFS.val/2)*2+1
        self.PFC = int(self.sPFC.val/2)*2+1
        self.MDS = int(self.sMDS.val)
        self.NOD = int(self.sNOD.val/16)*16
        self.TTH = int(self.sTTH.val)
        self.UR = int(self.sUR.val)
        self.SR = int(self.sSR.val)
        self.SPWS= int(self.sSPWS.val)
        if (loading_settings == 0):
            print ('Rebuilding depth map')
            disparity = self.stereo_depth_map(self.rectified_pair)
            self.dmObject.set_data(disparity)
            print ('Redraw depth map')
            plt.draw()

    def load_map_settings(self, event):
        loading_settings = 1
        fName = 'tuner/map_settings.txt'
        print('Loading parameters from file...')
        self.buttonl.label.set_text ("Loading...")
        f=open(fName, 'r')
        data = json.load(f)
        self.sSWS.set_val(data['SADWindowSize'])
        self.sPFS.set_val(data['preFilterSize'])
        self.sPFC.set_val(data['preFilterCap'])
        self.sMDS.set_val(data['minDisparity'])
        self.sNOD.set_val(data['numberOfDisparities'])
        self.sTTH.set_val(data['textureThreshold'])
        self.sUR.set_val(data['uniquenessRatio'])
        self.sSR.set_val(data['speckleRange'])
        self.sSPWS.set_val(data['speckleWindowSize'])
        f.close()
        self.buttonl.label.set_text ("Load settings")
        print ('Parameters loaded from file '+fName)
        print ('Redrawing depth map with loaded parameters...')
        loading_settings = 0
        self.update(0)
        print ('Done!')