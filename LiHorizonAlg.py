"""
Important note:
---------------
The case where no horizon is detected can be tricky. The following processes take place:
1) set the flag self.detected_hl_flag to logical True. This flag is used to decide whether next processes will be
performed.
2) assign np.nan to horizon parameters and detection latency. np.nan can be stored in numpy arrays of any np.dtype:
    self.det_position_hl, self.det_tilt_hl, self.theta, self.theta_deg, self.rho, self.latency
3) do not draw the horizon on the color frame/image to draw_and_save. Instead, put a text saying: "NO HORIZON IS
DETECTED"
"""

import cv2 as cv
from skimage.filters.rank import maximum as maxrank
import numpy as np
import os
from warnings import warn
from time import time


class LiF:
    def __init__(self, frame_t0=None, dsize=None, max_kernel_size=3):
        """
        A class implementing the horizon detection algorithm published in 'doi.org/10.1007/s11760-020-01733-0' by Fangxu
         Li et al.
        :param frame_tO: the first RGB frame of the video sequence. This frame is given once and gets updated by
        consecutive frames given by the user when detecting the horizon on the next frame frame_t1.
        (see self.get_horizon()).
        :param dsize: a tuple (width, height) indicating the resolution to which the image to process will be resized. If not
        given, the original image is processed without changing its resolution.
        :param max_kernel_size: the number of neighborhood to consider when computing the local maximum (see equation 8)
        ----------------------------------------------------------------------------------------------------------------
        IMPORTANT NOTES:
        A) this algorithm requires two consecutive frames to detect the horizon. Therefore:
            1) it cannot be applied on a single image captured at t0, unless the image captured at t0 - T is given,
            where T is very small (~30 ms to 50 ms)
            2) if a video file contains Nf frames, then only (Nf-1) horizons can be detected.
        B) This algorithm will be benchmarked on 1920x1080 images. Thus, parameters of the detected horizon will be
        computed to suit the original resolution (1920x1080), not the resolution given by the argument 'dsize'.
        ----------------------------------------------------------------------------------------------------------------
        """
        self.dsize = dsize
        self.frame_t0 = frame_t0
        self.frame_t1 = None
        if self.frame_t0 is None:
            warn("\nYou did not provide the argument frame_t0. If you intend invoking the method "
                 "self.get_horizon(), you must use the setter method self.set_frame_t0() to provide it. This is not "
                 "required if you want to invoke the method self.evaluate()", stacklevel=2)

        else:
            self._processes_t0()

        self.b = np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]], dtype=np.uint8)  # a square kernel with 4 connected neighborhoods. It is used in
        # the iterative erosion process (see equation 6)

        self.E_k0 = None  # the geodesically eroded image at the k-th iteration.
        self.E_k1 = None  # the geodesically eroded image at the (k+1)-th iteration. When convereged, it is the
        # clutter-filtered image self.P (see equation 2)
        self.P = None  # the clutter-filtered image (see equation 2), which is the geodesically eroded image at the last
        # iteration.
        self.max_kernel_size = max_kernel_size  # the number of neighborhood to consider when computing the local
        # maximum (see equation 8)
        self.max_kernel = np.ones((self.max_kernel_size, 1))  # the kernel used to compute the vertical local
        # max of the processed image self.E_k1

        # outputs
        self.det_position_hl = np.nan  # in pixels
        self.det_tilt_hl = np.nan  # in radians
        self.theta = np.nan  # in radians
        self.theta_deg = np.nan  # in degree
        self.rho = np.nan  # in pixels
        self.latency = np.nan  # in seconds
        self.img_with_hl = None  # the output image with the horizon line

        # flags
        self.detected_hl_flag = True

    def set_frame_t0(self, frame_t0):
        self.frame_t0 = frame_t0
        self._processes_t0()

    def set_dsize(self, dsize):
        self.dsize = dsize
        self._processes_dsize()

    def _processes_t0(self):
        """
        Executes required processes after setting the 'attribute self.frame_t0'.
        :return:
        """
        if len(self.frame_t0.shape) != 3:  # True if the frame does not have 3 channels
            raise Exception("the input frame_tO must be an BGR (or RGB) image")
        self.frame_t0 = cv.cvtColor(self.frame_t0, cv.COLOR_BGR2GRAY)
        self.org_height, self.org_width = self.frame_t0.shape
        self._processes_dsize()

    def _processes_dsize(self):
        """
        Executes processes related to setting the attribute self.dsize
        """
        if self.dsize is not None:  # True if the user wants to resize the image by providing the argument dsize
            self.resized_width, self.resized_height = self.dsize  # get the resized image width (self.resized_width)
            # and height (self.resized_height)
            self.width_multiplier = self.org_width / self.resized_width
            self.height_multiplier = self.org_height / self.resized_height
        else:
            self.resized_height, self.resized_width = self.org_height, self.org_width
            self.width_multiplier = 1
            self.height_multiplier = 1

    def get_horizon(self, frame_t1):
        """
        Get the position and tilt of the horizon line.
        :param frame_t1: video frame at instant t0 + T; where T is the time periode between adjacent frames. The horizon
        will be detected on this frame. Afterward, frame_t0 (see __init__()) will get updated by frame_t1.

        """
        self.starting_time = time()
        self.frame_t1 = frame_t1
        if self.frame_t0 is None:
            raise Exception("You did not provide the first frame 'self.frame_t0'. Use the setter method "
                            "self.set_frame_t0()")
        if len(self.frame_t1.shape) != 3:
            raise Exception("the input 'frame_t1' must be an BGR (or RGB) image")
        self.frame_colored = np.copy(self.frame_t1)
        self.frame_t1 = cv.cvtColor(self.frame_t1, cv.COLOR_BGR2GRAY)

        self.intraframe_difference()  # gets the edge response (using the intra-frame difference) stored in self.f
        self.morphology_hole_filling()  # apply the filtering process. The filtered image is 'self.E_k1'
        # cv.imwrite("reconstruction by erosion result.png", self.E_k1)
        self.candidates_extraction_and_fitting()
        self.frame_t0 = np.copy(self.frame_t1)
        # Rec = reconstruction(seed=self.f_m, mask=self.f, method='erosion', selem=self.b)
        self.R = np.uint8(np.multiply(self.R, 255 / np.max(self.R)))
        # cv.imwrite("local_maxima.png", self.R)
        self.get_edges()
        # cv.imwrite("edge.png", self.img_edges)
        self.fit_horizon()
        self.ending_time = time()
        self.latency = round((self.ending_time - self.starting_time), 4)
        self.draw_hl()
        # cv.imwrite("found horizon.png", self.img_with_hl)

    def intraframe_difference(self):
        if self.dsize is not None:
            self.frame_t0 = np.float32(cv.resize(src=self.frame_t0, dsize=self.dsize))
            self.frame_t1 = np.float32(cv.resize(src=self.frame_t1, dsize=self.dsize))
        else:
            self.frame_t0 = np.float32(self.frame_t0)
            self.frame_t1 = np.float32(self.frame_t1)
        self.f = np.uint8(np.abs(np.subtract(self.frame_t1, self.frame_t0)))  # conversion to
        # float is necessary. Otherwise, bit overflow takes place (e.g., 0 - 19 = 255 when subtracted values are
        # np.uint8)
        # cv.imwrite("f.png", self.f)

    def morphology_hole_filling(self):
        self.get_marker()  # gets a value for the marker attribute self.f_m
        # k0-th iteration of geodesic erosion, where k = 0 (in this line)
        self.E_k0 = self.geodesic_erosion(marker=self.f_m, mask=self.f, selem=self.b)
        # k1-th iteration of geodesic erosion (k1 = k0 + 1), which erodes the marker self.E_k0 using the structuring
        # element self.b constrained by the mask self.f
        self.E_k1 = self.geodesic_erosion(marker=self.E_k0, mask=self.f, selem=self.b)
        # after computation of self.E_k1, check if self.E_k1 = self.E_k0. If False, update self.E_k0 by self.E_k1 and
        # repeat the geodesic erosion to get a new value of self.E_k1. Otherwise, stop the process and the result
        # (i.e., filtered image) is self.E_k0.
        self.converged = np.all(np.equal(self.E_k0, self.E_k1))  # a boolean whose truth indicates convergence of the
        # reconstruction by erosion. Convergence requires that all pixels in self.E_k0 are exactly equal to self.E_k1
        key = ''
        while not self.converged and key != ord('s'):
            self.E_k0 = np.copy(self.E_k1)
            self.E_k1 = self.geodesic_erosion(marker=self.E_k0, mask=self.f, selem=self.b)
            self.converged = np.all(np.equal(self.E_k0, self.E_k1))
            # cv.imshow("Reconstruction by erosion steps", cv.resize(self.E_k1, dsize=(640, 480)))
            # key = cv.waitKey(2)

        # cv.destroyAllWindows()
        self.P = np.copy(self.E_k1)

    def candidates_extraction_and_fitting(self):
        # todo: get vertical local maximum X
        self.X = maxrank(image=self.P, selem=self.max_kernel)
        self.X = np.float16(self.X)
        # todo: get Xi1 and Xi2 by proper indexing of X
        self.Xi1 = self.X[0:self.resized_height - 1,
                   :]  # index from the first (0-th) row to the row before the last row.
        self.Xi2 = self.X[1:self.resized_height, :]  # index from the second (1-th) row to the last row
        # todo: get R
        self.R = np.abs(np.divide(np.subtract(self.Xi2, self.Xi1), self.Xi1))
        self.nan_inf_indexes = np.where(np.logical_or(np.isnan(self.R), np.isinf(self.R)))  # removing nan and infinite
        # numbers
        self.R[self.nan_inf_indexes] = 0

        # todo: get the image edges self.img_edges

    def geodesic_erosion(self, marker, mask, selem):
        eroded = cv.erode(src=marker, kernel=selem, iterations=1)
        return np.maximum(eroded, mask)

    def get_marker(self):
        """
        Gets the marker image self.f_m according to equation (3) of the paper
        :return:
        """
        self.I_max = np.max(self.f)
        # inner pixels of self.f_m(x, y) are equal to max(self.f) = self.I_max
        self.f_m = np.full_like(a=self.f, fill_value=self.I_max)

        # border pixels of self.f_m(x, y) are equal to self.f(x, y)
        self.f_m[0, :] = self.f[0, :]
        self.f_m[self.resized_height - 1, :] = self.f[self.resized_height - 1, :]
        self.f_m[:, 0] = self.f[:, 0]
        self.f_m[:, self.resized_width - 1] = self.f[:, self.resized_width - 1]

    def get_edges(self):
        self.img_edges = np.zeros(shape=self.R.shape, dtype=np.uint8)
        # self.R_ver_max = np.max(self.R, axis=0)  # get maximum values of R in the vertical direction

        self.R_ver_max_rows_index = np.argmax(self.R, axis=0)
        self.R_ver_max_cols_index = np.arange(0, self.resized_width)

        # self.R_ver_max_rows_index, self.R_ver_max_cols_index = np.where(self.R == self.R_ver_max)
        self.img_edges[self.R_ver_max_rows_index, self.R_ver_max_cols_index] = 255

    def fit_horizon(self):
        self.hough_lines = cv.HoughLines(image=self.img_edges, rho=1, theta=np.pi / 180, threshold=2)
        if self.hough_lines is not None:  # executes if Hough detects a line
            self.detected_hl_flag = True
            self.rho, self.theta = self.hough_lines[0][0]  # self.theta in radians
            self.det_tilt_hl = (
                    (np.pi / 2) - self.theta)  # I might be wrong about the sign. In such case, just invert the
            # sign
            self.det_position_hl = (self.rho - 0.5 * self.org_height * np.cos(self.theta)) / (np.sin(self.theta)) \
                                   * self.height_multiplier

            # converting to degrees
            self.det_tilt_hl = self.det_tilt_hl * (180 / np.pi)
            self.theta_deg = round(self.theta * (180 / np.pi), 4)

        else:
            self.detected_hl_flag = False
            self.rho = np.nan
            self.theta = np.nan
            self.det_position_hl = np.nan
            self.det_tilt_hl = np.nan
            self.theta_deg = np.nan
            self.latency = np.nan

    def draw_hl(self):
        """
        Draws the horizon line on attribute 'self.img_with_hl' if it is detected. Otherwise, the text 'NO HORIZON IS
        DETECTED' is put on the image.
        """
        self.img_with_hl = np.copy(self.frame_colored)
        if self.detected_hl_flag:
            self.xs_hl = int(0)

            self.xe_hl = int(self.resized_width)
            self.ys_hl = self.y_from_xpolar(self.xs_hl, self.rho, self.theta)
            self.ye_hl = self.y_from_xpolar(self.xe_hl, self.rho, self.theta)

            self.xe_hl = int(self.xe_hl * self.width_multiplier)
            self.ys_hl = int(self.ys_hl * self.height_multiplier)
            self.ye_hl = int(self.ye_hl * self.height_multiplier)

            cv.line(self.img_with_hl, (self.xs_hl, self.ys_hl), (self.xe_hl, self.ye_hl), (0, 0, 255), 5)
        else:
            put_text = "NO HORIZON IS DETECTED"
            org = (int(self.org_height / 2), int(self.org_height / 2))
            color = (0, 0, 255)
            cv.putText(img=self.img_with_hl, text=put_text, org=org, fontFace=0, fontScale=2, color=color, thickness=3)

    def y_from_xpolar(self, x, rho, theta):
        """
        returns y coordinate on the line defined by polar coordinates rho and theta, and that corresponds to x
        """
        return int((1 / np.sin(theta)) * (rho - x * np.cos(theta)))

    def evaluate(self, src_video_folder, src_gt_folder, dst_video_folder=r"", dst_quantitative_results_folder=r"",
                 draw_and_save=True):
        """
        Produces a .npy file containing quantitative results of the Horizon Edge Filter algorithm. The .npy file
        contains the following information for each image: |Y_gt - Y_det|, |alpha_gt - alpha_det|, and latency in
        seconds :param src_gt_folder: absolute path to the ground truth horizons corresponding to source video files.
        :param src_video_folder: absolute path to folder containing source video files to process :param
        dst_video_folder: absolute path where video files with drawn horizon will be saved. :param
        dst_quantitative_results_folder: destination folder where quantitative results will be saved. :param
        draw_and_save: if True, all detected horizons will be drawn on their corresponding frames and saved as video
        files in the folder specified by 'dst_video_folder'.
        """
        src_video_names = sorted(os.listdir(src_video_folder))
        srt_gt_names = sorted(os.listdir(src_gt_folder))
        for src_video_name, src_gt_name in zip(src_video_names, srt_gt_names):
            print("{} will correspond to {}".format(src_video_name, src_gt_name))

        # Allowing the user to verify that each gt .npy file corresponds to the correct video file # # # # # # # # # # #
        while True:
            yn = input("Above are the video files and their corresponding gt files. If they are correct, click on 'y'"
                       " to proceed, otherwise, click on 'n'.\n"
                       "If one or more video file has incorrect gt file correspondence, we recommend to rename the"
                       "files with similar names.")
            if yn == 'y':
                break
            elif yn == 'n':
                print("\nTHE QUANTITATIVE EVALUATION IS ABORTED AS ONE OR MORE LOADED GT FILES DOES NOT CORRESPOND TO "
                      "THE CORRECT VIDEO FILE")
                return
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        self.det_horizons_all_files = np.empty(shape=[0, 5])
        nbr_of_vids = len(src_video_names)
        vid_indx = 0
        for src_video_name, src_gt_name in zip(src_video_names, srt_gt_names):  # each iteration processes one video
            # file
            vid_indx += 1
            print("loaded video/loaded gt: {}/{}".format(src_video_name, src_gt_name))  # printing which video file
            # correspond to which gt file

            src_video_path = os.path.join(src_video_folder, src_video_name)
            src_gt_path = os.path.join(src_gt_folder, src_gt_name)

            cap = cv.VideoCapture(src_video_path)  # create a video reader object
            # Creating the video writer # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            fps = cap.get(propId=cv.CAP_PROP_FPS)
            self.org_width = int(cap.get(propId=cv.CAP_PROP_FRAME_WIDTH))
            self.org_height = int(cap.get(propId=cv.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')  # codec used to compress the video.
            if draw_and_save:
                dst_vid_path = os.path.join(dst_video_folder, "F.Li_" + src_video_name)
                video_writer = cv.VideoWriter(dst_vid_path, fourcc, fps, (self.org_width, self.org_height),
                                              True)  # video writer object
            self.gt_horizons = np.load(src_gt_path)
            nbr_of_annotations = self.gt_horizons.shape[0]
            nbr_of_frames = int(cap.get(propId=cv.CAP_PROP_FRAME_COUNT))

            if nbr_of_frames != nbr_of_annotations:
                error_text_1 = "WARNING: The number of annotations (={}) does not match the number of frames (={})". \
                    format(nbr_of_annotations, nbr_of_frames)
                print(error_text_1)

            no_error_flag, frame = cap.read()
            self.set_frame_t0(frame_t0=frame)

            if not no_error_flag:
                raise Exception("The first frame of the video file specified by the path {} could not be read".
                                format(src_video_path))

            self.det_horizons_per_file = np.zeros((nbr_of_annotations, 5))
            for idx, gt_horizon in enumerate(self.gt_horizons):
                no_error_flag, frame = cap.read()
                if not no_error_flag:
                    break
                self.get_horizon(frame_t1=frame)  # gets the horizon
                # position and tilt
                self.gt_position_hl, self.gt_tilt_hl = gt_horizon[0], gt_horizon[1]
                # print("detected position/gt position {}/{};\n detected tilt/gt tilt {}/{}".
                #       format(self.det_position_hl, self.gt_position_hl, self.det_tilt_hl, self.gt_tilt_hl))
                # print("Latency = {} seconds".format(self.latency))
                print("Frame {}/{}. Video {}/{}".format(idx, nbr_of_frames, vid_indx, nbr_of_vids))
                self.det_horizons_per_file[idx] = [self.det_position_hl,
                                                   self.det_tilt_hl,
                                                   round(abs(self.det_position_hl - self.gt_position_hl), 4),
                                                   round(abs(self.det_tilt_hl - self.gt_tilt_hl), 4),
                                                   self.latency]
                # todo: draw the horizon line only if it is detected (use a flag at line # 382) and add an else to set
                #  it to the NOT state
                self.draw_hl()  # draws the horizon on self.img_with_hl
                video_writer.write(self.img_with_hl)
            cap.release()
            video_writer.release()
            print("The video file {} has been processed.".format(src_video_name))
            src_video_name_no_ext = os.path.splitext(src_video_name)[0]
            det_horizons_per_file_dst_path = os.path.join(dst_quantitative_results_folder,
                                                          src_video_name_no_ext + ".npy")
            np.save(det_horizons_per_file_dst_path, self.det_horizons_per_file)  # save detected horizons of the current
            # video file
            self.det_horizons_all_files = np.append(self.det_horizons_all_files,
                                                    self.det_horizons_per_file,
                                                    axis=0)
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # after processing all video files, save quantitative results as .npy file
        src_video_folder_name = os.path.basename(src_video_folder)
        dst_detected_path = os.path.join(dst_quantitative_results_folder,
                                         "all_det_hl_" + src_video_folder_name + ".npy")
        np.save(dst_detected_path, self.det_horizons_all_files)

        # temporarcy portion of code
        loaded_detected = np.load(dst_detected_path)
        detected_mean = np.mean(loaded_detected, axis=0)
        detected_50p = np.percentile(loaded_detected, q=50, axis=0)
        print("mean: Y error = {} px, angle error = {} deg, latency = {} s".format(detected_mean[2],
                                                                                   detected_mean[3],
                                                                                   detected_mean[4]))
        print("median: Y error = {} px, angle error = {} deg, latency = {} s".format(detected_50p[2],
                                                                                     detected_50p[3],
                                                                                     detected_50p[4]))
