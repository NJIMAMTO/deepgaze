import numpy as np
import os
import tensorflow as tf
import cv2
from deepgaze.head_pose_estimation import CnnHeadPoseEstimator

sess = tf.Session() #Launch the graph in a session.
my_head_pose_estimator = CnnHeadPoseEstimator(sess) #Head pose estimation object
my_head_pose_estimator.load_pitch_variables("../../etc/tensorflow/head_pose/pitch/cnn_cccdd_30k.tf")
my_head_pose_estimator.load_yaw_variables("../../etc/tensorflow/head_pose/yaw/cnn_cccdd_30k.tf")
my_head_pose_estimator.load_roll_variables("../../etc/tensorflow/head_pose/roll/cnn_cccdd_30k.tf")

#Function used to get the rotation matrix
def rot2rotmat(yaw,pitch,roll):
    x = roll
    y = pitch
    z = yaw
    ch = np.cos(z)
    sh = np.sin(z)
    ca = np.cos(y)
    sa = np.sin(y)
    cb = np.cos(x)
    sb = np.sin(x)
    rot = np.zeros((3,3), 'float32')
    rot[0][0] = ch * ca
    rot[0][1] = sh*sb - ch*sa*cb
    rot[0][2] = ch*sa*sb + sh*cb
    rot[1][0] = sa
    rot[1][1] = ca * cb
    rot[1][2] = -ca * sb
    rot[2][0] = -sh * ca
    rot[2][1] = sh*sa*cb + ch*sb
    rot[2][2] = -sh*sa*sb + ch*cb
    return rot

#Defining the video capture object
video_capture = cv2.VideoCapture(0)

if(video_capture.isOpened() == False):
    print("Error: the resource is busy or unvailable")
else:
    print("The video source has been opened correctly...")

#Create the main window and move it
cv2.namedWindow('Video')
cv2.moveWindow('Video', 500, 50)

#Obtaining the CAM dimension
#cam_w = int(video_capture.get(3))
#cam_h = int(video_capture.get(4))

#print(str(cam_w)+str(cam_h))

while(True):

    ret, frame = video_capture.read()
    height, width, channels = frame.shape

    clp_frame = frame[0:height, (width - height) * 1/2 : (width + height) * 1/2]   

    c_x = height / 2
    c_y = height / 2
    f_x = c_x / np.tan(60/2 * np.pi / 180)
    f_y = f_x
    camera_matrix = np.float32([[f_x, 0.0, c_x],
                                [0.0, f_y, c_y], 
                                [0.0, 0.0, 1.0] ])
    #print("Estimated camera matrix: \n" + str(camera_matrix) + "\n")

    #Distortion coefficients
    camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])
    #Defining the axes
    axis = np.float32([[0.0, 0.0, 0.0], 
                       [0.0, 0.0, 0.0], 
                       [0.0, 0.0, 0.5]])

    #Evaluate the pitch,yaw and roll angle using a CNN               
    pitch = my_head_pose_estimator.return_pitch(clp_frame)
    yaw = my_head_pose_estimator.return_yaw(clp_frame)
    roll = my_head_pose_estimator.return_roll(clp_frame)

    #print("[pitch]" + str(pitch[0,0,0]) + "[yaw]" + str(yaw[0,0,0]) + "[roll]" + str(roll[0,0,0]))
    #Getting rotation and translation vector
    rot_matrix = rot2rotmat(-yaw[0,0,0],pitch[0,0,0],roll[0,0,0]) #Deepgaze use different convention for the Yaw, we have to use the minus sign

    #Attention: OpenCV uses a right-handed coordinates system:
    #Looking along optical axis of the camera, X goes right, Y goes downward and Z goes forward.
    rvec, jacobian = cv2.Rodrigues(rot_matrix)
    tvec = np.array([0.0, 0.0, 1.0], np.float) # translation vector
    #print rvec

    imgpts, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrix, camera_distortion)
    p_start = (int(c_x), int(c_y))
    p_stop = (int(imgpts[2][0][0]), int(imgpts[2][0][1]))
    #print("point start: " + str(p_start))
    #print("point stop: " + str(p_stop))
    #print("")

    #cv2.line(clp_frame, p_start, p_stop, (0,0,255), 3) #RED
    #cv2.circle(clp_frame, p_start, 1, (0,255,0), 3) #GREEN

    cv2.putText(clp_frame, "yaw"   + str(yaw[0,0,0])   + "[deg]", (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1);
    cv2.putText(clp_frame, "pitch" + str(pitch[0,0,0]) + "[deg]", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1);
    cv2.putText(clp_frame, "roll " + str(roll[0,0,0])  + "[deg]", (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1);

    #test
    cv2.rectangle(clp_frame, (height/2 - 150, height/2 - 150), (height/2 + 150, height/2 + 150), (0, 0, 255), 5)
    cv2.imshow('Video', clp_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

#Release the camera
video_capture.release()
print("Bye...")

