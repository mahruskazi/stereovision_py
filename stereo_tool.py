import argparse
import sys
import cv2
import time

from capture.capture_images import CaptureImages
from calibrate.calibrate_cameras import CalibrateCameras
from tuner.tuner import Tuner
from tracking.tracking import Tracking

class CLI():
    def __init__(self):
        parser = argparse.ArgumentParser(
                description='A CLI to help calibrate stereovision, calculate distance from a depth map, and track an object',
                usage='''stereovision_py <command> [<args>]

                        Commands:
                        capture         Capture pairs of images to then be used for stereovision calibration
                        calibrate       Calibrate the cameras to create a rectfied image pair
                        tune            Tune the stereovision depth map
                        tracker         Track a object in real time and measure the distance to it
                        ''')
        parser.add_argument('command', help='Subcommand to run')

        args = parser.parse_args(sys.argv[1:2])
        command = '_' + args.command
        if not hasattr(self, command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)

        getattr(self, command)()

    def _capture(self):
        parser = argparse.ArgumentParser(description='Capture pairs of images to then be used for stereovision calibration')
        parser.add_argument('-i', '--interval', help='How long to wait before saving the next image', type=int, default=5)

        args = parser.parse_args(sys.argv[2:])

        cap = CaptureImages()
        cap.setup_capture()
        cap.save_images(args.interval)

        cap.release_cameras()

    def _calibrate(self):
        parser = argparse.ArgumentParser(description='Calibrate the cameras to create a rectfied image pair')
        parser.add_argument('-p', '--photos', help='Number of pairs of images to use', type=int, required=True)
        parser.add_argument('--height', help='Height of the calibration images', type=int, default=480)
        parser.add_argument('--width', help='Width of the calibration images', type=int, default=640)
        parser.add_argument('--rows', help='Number of rows on the Chessboard', type=int, default=6)
        parser.add_argument('--columns', help='Number of columns on the Chessboard', type=int, default=9)
        parser.add_argument('--size', help='Size of the squares on the Chessboard', type=float, default=2.5)

        args = parser.parse_args(sys.argv[2:])

        cal = CalibrateCameras()

        cal.total_photos = args.photos
        cal.img_height = args.height
        cal.img_width = args.width

        cal.rows = args.rows
        cal.columns = args.columns
        cal.square_size = args.size

        cal.start_calibration()

    def _tune(self):
        parser = argparse.ArgumentParser(description='Tune the stereovision depth map')
        parser.add_argument('--save_scene', help='Number of pairs of images to use', action='store_true')
        args = parser.parse_args(sys.argv[2:])

        cap = CaptureImages()
        cap.setup_capture()

        if args.save_scene:
            cap.capture_scene()
        else:
            Tuner()

        cap.release_cameras()

    def _tracker(self):
        parser = argparse.ArgumentParser(description='Track a object in real time and measure the distance to it')
        args = parser.parse_args(sys.argv[2:])

        cap = CaptureImages()
        cap.setup_capture()
        Tracking(cap.left_cap, cap.right_cap)


if __name__ == "__main__":
    CLI()