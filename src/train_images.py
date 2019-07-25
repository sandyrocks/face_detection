import cv2
import pickle
import fire
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from align import AlignDlib
from model import create_model
# classifer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.externals import joblib 
from sklearn.metrics import f1_score, accuracy_score


class IdentityMetadata():
	def __init__(self, base, name, file):
		# dataset base directory
		self.base = base
		# identity name
		self.name = name
		# image file name
		self.file = file

	def __repr__(self):
		return self.image_path()

	def image_path(self):
		return os.path.join(self.base, self.name, self.file) 



class Trainer():
	def __init__(self):
		# self.image_dir = sys.argv[1]
		# self.image_dir = "/home/webwerks/projects/project-face-recognition/23-07/face_detection/uploads"
		self.image_dir = "/home/webwerks/projects/project-face-recognition/23-07/face_detection/src/images"
		self.model = self.create_model()
		self.alignment = AlignDlib('models/landmarks.dat')


	def create_model(self):
		nn4_small2_pretrained = create_model()
		nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')
		return nn4_small2_pretrained

	def load_metadata(self, path):
		print("loading metadata ...")
		metadata = []
		for i in sorted(os.listdir(path)):
			for f in sorted(os.listdir(os.path.join(path, i))):
				# Check file extension. Allow only jpg/jpeg' files.
				ext = os.path.splitext(f)[1].lower()
				if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
					metadata.append(IdentityMetadata(path, i, f))
		return np.array(metadata)

	def load_image(self, path):
		img = cv2.imread(path, 1)
		return img


	def align_image(self, img):
		return  self.alignment.align(96, img, self.alignment.getLargestFaceBoundingBox(img), 
							   landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
		

	def train(self):
		metadata = self.load_metadata(self.image_dir)
		np.save('train-files/metadata.npy', metadata)
		
		print("creating embeddings ...")
		embedded = {}

		for i, m in enumerate(metadata):
			
			img = self.load_image(m.image_path())
			img = self.align_image(img)
			
			if img is not None:
				# scale RGB values to interval [0,1]
				img = (img / 255.).astype(np.float32)
				# obtain embedding vector for image
				embedded[m.image_path()] = self.model.predict(np.expand_dims(img, axis=0))[0] 

		pickle.dump( embedded, open("train-files/embeddings.pkl", "wb" ) )


	def train_classifier(self):
		print("training classifier ...")

		embedded = pickle.load(open("train-files/embeddings.pkl", "rb" ))
		metadata = np.load('train-files/metadata.npy')

		targets = np.array([m.name for m in metadata])
		encoder = LabelEncoder()
		encoder.fit(targets)
		pickle.dump( encoder, open("train-files/encoder.pkl", "wb" ) )
		# Numerical encoding of identities
		y = encoder.transform(targets)

		train_idx = np.arange(metadata.shape[0]) % 2 != 0
		test_idx = np.arange(metadata.shape[0]) % 2 == 0

		
		X_train = embedded[train_idx]
		X_test = embedded[test_idx]

		y_train = y[train_idx]
		y_test = y[test_idx]
		
		svc = LinearSVC()
		svc.fit(X_train, y_train)

		joblib.dump(svc, 'train-files/svc.pkl')
		
		acc_svc = accuracy_score(y_test, svc.predict(X_test))

		print(f'SVM accuracy = {acc_svc}')
		

if __name__ == '__main__':
	train_obj = Trainer()
	train_obj.train()
	# train_obj.train_classifier()
	# fire.Fire({
	#     'train': train
	# })