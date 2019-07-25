import pickle
import cv2
import numpy as np
from align import AlignDlib
from sklearn.externals import joblib
from train_images import Trainer
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class StaticImages:
	def __init__(self):
		self.trainer_obj = Trainer()
		self.embedded = pickle.load(open("train-files/embeddings.pkl", "rb" ))
		# self.metadata = np.load('train-files/metadata.npy')
		# self.encoder = pickle.load(open("train-files/encoder.pkl", "rb" ))
		# self.svc = joblib.load('train-files/svc.pkl')
		self.color = (255, 0, 0) #BGR 0-255 
		self.stroke = 2
		self.required_size = (224, 224)
		self.font = cv2.FONT_HERSHEY_SIMPLEX
		self.threshold = 0.65


	def distance(self, emb1, emb2):
		return np.sum(np.square(emb1 - emb2))

	def face_distance(self, face_encodings, face_to_compare):
		"""
		Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
		for each comparison face. The distance tells you how similar the faces are.
		:param faces: List of face encodings to compare
		:param face_to_compare: A face encoding to compare against
		:return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
		"""
		# if len(face_encodings) == 0:
		#     return np.empty((0))
		# np.linalg.norm(face_encodings - face_to_compare, axis=1)
		# dist = []
		# for each in face_encodings:
		# 	dist.append(self.distance(each,  face_to_compare))

		return np.linalg.norm(face_encodings - face_to_compare, axis=1)

	def test(self, img_path):
		
		frame = cv2.imread(img_path, 1)
		faces = self.trainer_obj.alignment.getAllFaceBoundingBoxes(frame)

		
		print(faces)
		for face in faces:
			frame_aligned = self.trainer_obj.alignment.align(96, frame, face, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
			frame_aligned = (frame_aligned / 255.).astype(np.float32)
			pred_embedded = self.trainer_obj.model.predict(np.expand_dims(frame_aligned, axis=0))[0] 
			names = [name.split("/")[-2] for name in list(self.embedded.keys())]
			dist = self.face_distance(list(self.embedded.values()), pred_embedded)
			name = names[np.argmin(dist)] if any(x <= self.threshold for x in dist) else "unknown"
			print(min(dist), name)
			x1, y1, x2, y2 = face.left(), face.top() , face.left() + face.width(),face.top() + face.height()
			
			cv2.rectangle(frame,(x1, y1), (x2, y2), self.color, self.stroke)
			cv2.putText(frame, name, (x1,y1), self.font, 1, self.color, self.stroke, cv2.LINE_AA)


		frame = cv2.resize(frame, (1080, 920)) 	
		# Display the resulting frame
		cv2.imshow('frame_org',frame)	
	
		
		k = cv2.waitKey(0)
		if k == 27:         # wait for ESC key to exit
		    cv2.destroyAllWindows()
		elif k == ord('s'): # wait for 's' key to save and exit
		    # cv2.imwrite('output.png',frame)
		    cv2.destroyAllWindows()
				
		
if __name__ == '__main__':
	static_img_obj = StaticImages()
	img_path = "/home/webwerks/projects/project-face-recognition/23-07/pats/grp_test/DSC07296.JPG"
	# img_path = "/home/webwerks/projects/project-face-recognition/23-07/pats/grp_test/group.jpg"
	static_img_obj.test(img_path)
			