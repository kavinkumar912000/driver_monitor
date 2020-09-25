# importing the necessary packages
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
from threading import Thread
import imutils
import playsound
import dlib
import queue
import cv2
import time
import mediapipe as mp
import math
from datetime import datetime
from datetime import date
ar=0
al=0
r=0
u=0
ar=0
al=0
sound_path="alarm.wav"
threadStatusQ = queue.Queue()
ALARM_ON = False


#find left and right movement 
#calculating eye aspect ratio
def eye_aspect_ratio(eye):
	# compute the euclidean distances between the vertical
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	return ear

#calculating mouth aspect ratio
def mouth_aspect_ratio(mou):
	# compute the euclidean distances between the horizontal
	X   = dist.euclidean(mou[0], mou[6])
	# compute the euclidean distances between the vertical
	Y1  = dist.euclidean(mou[2], mou[10])
	Y2  = dist.euclidean(mou[4], mou[8])
	# taking average
	Y   = (Y1+Y2)/2.0
	# compute mouth aspect ratio
	mar = Y/X
	return mar

def soundAlert(path, threadStatusQ):
	with open('sleep.csv', 'a') as file:
		now = datetime.now()
		dtString = now.strftime('%H:%M:%S')
		today = date.today()
		dtdate = today.strftime("%d/%m/%Y")
		file.writelines(f'\n{"Sleep Alert"},{dtString},{dtdate}')
	playsound.playsound(path)
    #while True:
    #    if not threadStatusQ.empty():
    #        FINISHED = threadStatusQ.get()
    #        if FINISHED:
    #            break


camera = cv2.VideoCapture("VIDEO_20200918_181446282.mp4")
pose_tracker = mp.examples.UpperBodyPoseTracker()
predictor_path = 'shape_predictor_68_face_landmarks.dat'
if not camera.isOpened():
    raise IOError("Cannot open webcam")

# define constants for aspect ratios
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 30
MOU_AR_THRESH = 0.65

COUNTER = 0
yawnStatus = False
yawns = 0
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# grab the indexes of the facial landmarks for the left and right eye
# also for the mouth
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# loop over captuing video
while True:
	# grab the frame from the camera, resize
	# it, and convert it to grayscale
	# channels)
	ret, frame = camera.read()
	frame = imutils.resize(frame, width=940)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	prev_yawn_status = yawnStatus
	if(u==0):
		print("look at the camera and be erect ...")
		time.sleep(2)
		cv2.imwrite("img.jpg", frame)
		u=u+1
		pose_landmarks, annotated_image = pose_tracker.run(input_file='img.jpg')
	else:
		cv2.imwrite("img1.jpg",frame)
		pose_landmarks, annotated_image = pose_tracker.run(input_file='img1.jpg')
	if pose_landmarks is None:
		print("person not found")
		continue
	keypoints = []
	for data_point in pose_landmarks.landmark:
		keypoints.append({'X': data_point.x,'Y': data_point.y,'Z': data_point.z,'Visibility': data_point.visibility,})
	x = keypoints[0]
	y = keypoints[11]
	a = x['X'] - y['X']
	b = x['Y'] - y['Y']
	c = x['Z'] - y['Z']
	d = math.sqrt(a * a + b * b)
	d=d*100
	x1=keypoints[0]
	y1=keypoints[12]
	a1=x1['X']-y1['X']
	b1=x1['Y']-y1['Y']
	c1 = x1['Z'] - y1['Z']
	d1=math.sqrt(a1*a1 +b1*b1)
	d1=d1*100
	with open('sleep.csv', 'a') as file:
		now = datetime.now()
		dtString = now.strftime('%H:%M:%S')
		today = date.today()
		dtdate = today.strftime("%d/%m/%Y")
		if(u==1):
			ar=d
			al=d1
			u=u+1
		else:
			if ar-2 > d :
				print("Turned Left")
				file.writelines(f'\n{"Turned left"},{dtString},{dtdate}')
			elif al-2 > d1 :
				print("Turned Right")
				file.writelines(f'\n{"Turned Right"},{dtString},{dtdate}')
			else:
				print("A T T E N T I O N")
		if(d>ar and d1>al):
			print("head is facing up")
			file.writelines(f'\n{"Facing up"},{dtString},{dtdate}')
		elif(d<ar and d1<al):
			print("head is facing down")
			file.writelines(f'\n{"Facing down"},{dtString},{dtdate}')
		else:
			print("")



	# detect faces in the grayscale frame
	rects = detector(gray, 0)
	#determining left and right 

	
	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		mouth = shape[mStart:mEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		mouEAR = mouth_aspect_ratio(mouth)
		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		mouthHull = cv2.convexHull(mouth)
		#cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 255), 1)
		#cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 255), 1)
		#cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if ear < EYE_AR_THRESH:
			COUNTER += 1
			cv2.putText(frame, "Eyes Closed ", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

			# if the eyes were closed for a sufficient number of
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				# draw an alarm on the frame
				cv2.putText(frame, "DROWSINESS ALERT!", (10, 50),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				if not ALARM_ON:
					ALARM_ON = True
					threadStatusQ.put(not ALARM_ON)
					thread = Thread(target=soundAlert, args=(sound_path, threadStatusQ,))
					thread.setDaemon(True)
					thread.start()
			else:
				ALARM_ON=False




		# otherwise, the eye aspect ratio is not below the blink
		# threshold, so reset the counter and alarm
		else:
			COUNTER = 0
			cv2.putText(frame, "Eyes Open ", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

		#cv2.putText(frame, "EAR: {:.2f}".format(ear), (480, 30),
		#	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# yawning detections

		if mouEAR > MOU_AR_THRESH:
			cv2.putText(frame, "Yawning ", (10, 70),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			yawnStatus = True
			output_text = "Yawn Count: " + str(yawns + 1)
			cv2.putText(frame, output_text, (10,100),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,0,0),2)
			with open('sleep.csv', 'a') as file:
				now = datetime.now()
				dtString = now.strftime('%H:%M:%S')
				today = date.today()
				dtdate = today.strftime("%d/%m/%Y")
				file.writelines(f'\n{"Yawning"},{dtString},{dtdate}')
		else:
			yawnStatus = False

		if prev_yawn_status == True and yawnStatus == False:
			yawns+=1

		#cv2.putText(frame, "MAR: {:.2f}".format(mouEAR), (480, 60),
		#	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		#cv2.putText(frame,"...",(370,470),cv2.FONT_HERSHEY_COMPLEX,0.6,(153,51,102),1)
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
camera.release()
