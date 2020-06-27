## Advanced_Lane_Finding

## Goal
In this project, your goal is to write a software pipeline to identify the lane boundaries in a video

## Writeup for Project


---

**Advanced Lane Finding Project**

The steps taken in this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  If you want to extract more test images from the videos, you can simply use an image writing method like `cv2.imwrite()`, i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.  

To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `output_images`, and include a description in your writeup for the project of what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another optional challenge and is brutal!

## Project Overview

In order to detect the lane lines in a video stream we must accomplish the folowing:

- **Camera Calibration** - Calibrate the camera to correct for image distortions. For this I used a set of chessboard images, knowing the distance and angles between common features like corners, one can calculate the tranformation functions and apply them to the video frames.

- **Color Transform** - I used a set of image manipulation techniques to accentuate certain features like lane lines. Then I used color space transformations, like from RGB to HLS, channel separation, like separating the S channel from the HLS image and image gradient to allow us to identify the desired lines.

- **Perspective Transform** - I applied a "bird’s-eye view transform" that let me view a lane from above and thus identify the lane lines, measure its curvature and respective radius.

- **Lane Pixel Detection** - After that I analysed the transformed image and try to detect the lane pixels. I used a series of windows and identified the lane lines by finding the peeks in a histogram of each window's.

- **Image augmentation** - I added a series of overlays to the image to: identify the lane lines, showed the "bird's eye view" perspective, showed the location of the rectangle windows where the lane pixels are and finaly metrics on the radius of curvature and distance to the center of the road.

- **Pipeline** - And finally put it all together in a pipeline so we can apply it to the video stream.

### Camera Calibration

I started by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0 (depth), such that the object points are the same for each calibration image.  

I used the OpenCV functions **findChessboardCorners** and **drawChessboardCorners** to identify the locations of corners on a chessboard photos in **camera_cal** folder taken from different angles.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

```
# Displaying the corners
        cv2.drawChessboardCorners(img, (chess_columns,chess_rows), corners, ret)
        f, (axis1, axis2) = plt.subplots(1, 2, figsize=(15,10))
        axis1.imshow(cv2.cvtColor(mplimg.imread(cal_image), cv2.COLOR_BGR2RGB))
        axis1.set_title(str(i)+'. Input Image ('+image_name+'.jpg)', fontsize=16)
        axis2.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axis2.set_title(str(i)+'. Image with Corners ('+image_name+'_out.jpg)', fontsize=16)
        plt.savefig(chess_distortion_correction_folder+"/"+image_name+'_out.jpg')

```

Finally I calibrated the camera and got my matrix and distortion

Applied a distortion correction to raw images placed in folder **test_images**.

- Input : calculated camera calibration matrix and distortion coefficients to remove distortion from an image, and

- Output : the undistorted image.

### Color Transform
Here I used a series of image manipulation techniquest to detect the edges, lines, of an image
  - Converted the warped image to different color spaces and created binary thresholded images which highlight only lane lines and ignore everything else.
  - Following color channels were used:
    - 'S' Channel from HLS color space, with a minimum threshold = 180 & max threshold = 255
    - 'L' Channel from LUV color space, with a min threshold = 225 & max threshold = 255,
    - 'B' channel from the LAB color space, with a min threshold = 155 & max threshold = 200,
   - Created a combined binary threshold based on the above three mentioned binary thresholds.

### Perspective Transform
For writing my perspective transform I defined a matrix of source and destination points in the images as to transform them to the "bird's eye view" using OpenCV getPerspectiveTransform function. It takes as inputs an image (`img`) and hardcodes the source (`src`) and destination (`dst`) points.  
- It uses the CV2's **getPerspectiveTransform()** and **warpPerspective()** fns and **undistort()** written as discussed above.


### Lane Pixel Detection

The next challenge is, by using the transformed image, identify the lane line pixels. To accomplish this I used a method called "Peaks in a Histogram" where I analysed the histogram of section of the image, window, and identify the peaks which represent the location of the lane lines.

Starting with the combined binary image to isolate only the pixels belonging to lane lines, I fit the polynomial to each lane line, as follows:

- Identified peaks in a histogram of the image to determine location of lane lines.
- Identified all non-zero pixels around histogram peaks using the numpy function numpy.nonzero().
- Fitted polynomial to each lane using the numpy's fn. numpy.polyfit().

With this, I was able to calculate the position of the vehicle w.r.t center with the following calculations:

- Calculated x intercepts avg. from each of the two polynomials position
- Calculated distance from center by taking the abs value of the vehicle's position and subtracting the halfway point along the horizontal axis distance from center.
- If horizontal position of the car was > image_width/2, then car was considered to be on left of center, else right of center.
- Finally, the center distance was converted from pixels to meters by multiplying the number of pixels by 3.7/700.


### Image Augmentation

```
# Find radius of curvature for both lane line
xm_per_pix = 3.7/700 # meteres/pixel in x dimension
ym_per_pix = 30.0/720 # meters/pixel in y dimension

left_lane_fit_curvature = np.polyfit(left_y*ym_per_pix, left_x*xm_per_pix, 2)
right_lane_fit_curvature = np.polyfit(right_y*ym_per_pix, right_x*xm_per_pix, 2)
radius_left_curve = ((1 + (2*left_lane_fit_curvature[0]*np.max(left_y) + left_lane_fit_curvature[1])**2)**1.5) \
                             /np.absolute(2*left_lane_fit_curvature[0])
radius_right_curve = ((1 + (2*right_lane_fit_curvature[0]*np.max(left_y) + right_lane_fit_curvature[1])**2)**1.5) \
                                /np.absolute(2*right_lane_fit_curvature[0])

```
    
    # Find position of the vehicle
    vehicles_center = abs(640 - ((right_x_int+left_x_int)/2))
  ```  
    
```
# Calculating the vehicle position relative to the center of the lane
    position = (right_x_int + left_x_int)/2
    distance_from_center = abs((640 - position)*3.7/700) 
                
    Minv = cv2.getPerspectiveTransform(destination, source)
    
    warped_zero = np.zeros_like(combined_binary_img).astype(np.uint8)
    color_warped = np.dstack((warped_zero, warped_zero, warped_zero))
    pts_left = np.array([np.flipud(np.transpose(np.vstack([Left.fitx, Left.Y])))])
    pts_right = np.array([np.transpose(np.vstack([right_fitx, Right.Y]))])
    pts = np.hstack((pts_left, pts_right))
    cv2.polylines(color_warped, np.int_([pts]), isClosed=False, color=(0,0,255), thickness = 40)
    cv2.fillPoly(color_warped, np.int_(pts), (34,255,34))
    new_warped = cv2.warpPerspective(color_warped, Minv, (image.shape[1], image.shape[0]))
    result = cv2.addWeighted(undistorted_image, 1, new_warped, 0.5, 0)
    
    ```
 
 ### Pipeline
 
 This is the pipeline for the video
 ```
 # o/p folder creation for Chess 'camera_cal' dir i/p images
output_videos_folder = 'output_videos'
if not os.path.exists(output_videos_folder):
    os.makedirs(output_videos_folder)
 ```
 ### Discussion

- The pipeline developed in this project did a real good job in detecting the lane lines for the  [project_video.mp4] video, which implied that the code worrks well for the known ideal conditions having distinct lane lines, and with not much shadows.

- Code mostly did fine on the [challenge_result.mp4] video, with sometimes losing lane lines when heavy shadow was there.

- Code failed miserably on the [harder_challenge_video.mp4] video.


### ReadMe
For great writeup Check out the [writeup template](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md)
