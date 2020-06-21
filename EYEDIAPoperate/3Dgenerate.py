import face_alignment
import os
from skimage import io
import cv2


#type = ['S', 'M']
type = ['M']
my_path = "/disks/disk2/zhangbocheng/eyediap/EYEDIAP"

for t in type:
	# Session selection (from EYEDIAP scripts)
	sessions = []
	for P in range(1,17):
	    if P < 12 or P > 13:
	        sessions.append(str(P) + "_A_CS_" + t)

	frames_done = 0


	for session in sessions:
		print("frames_done", frames_done)
		print("Session: ", session)
		# NOTE: this may not work depending on session. Change accordingly.
		#session_str = get_session_string(session_num)


		session_str = session



		landmarks_file = os.path.join('/disks/disk2/zhangbocheng/landmarks', ('result3D_' + session_str + '.txt'))
		datafile = os.path.join(my_path, 'Data', session_str, 'rgb_vga.mov')

		vc = cv2.VideoCapture(datafile)
		c=0
		rval=vc.isOpened()

		while rval:   #循环读取视频帧
			c = c + 1
			if (c - 1) % 10 == 0:
				rval, frame = vc.read()
				if rval:
					fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda', flip_input=False)
					#input = io.imread(frame)
					preds = fa.get_landmarks(frame)
					#print(preds)
					with open(landmarks_file, 'a') as infile:
						#infile.write(preds)
						#for i in range(len(preds)):
							#s = str(preds[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
							#s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
							#infile.write(s)
						infile.write(str(preds))
					print("operate the", c)
					cv2.waitKey(1)
				else:
				    break
		    
		#fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)
		#fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda', flip_input=False)   	
		#input = io.imread(datafile)
		#preds = fa.get_landmarks(input)
		#print(preds)

		#with open(landmarks_file, 'wb') as infile:
			#infile.write(preds)

		