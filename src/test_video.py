import pickle
import cv2
import numpy as np
from align import AlignDlib
from sklearn.externals import joblib
from train_images import Trainer
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import datetime

class Videos:
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
        self.threshold = 0.5


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
        #   dist.append(self import time.distance(each,  face_to_compare))

        return np.linalg.norm(face_encodings - face_to_compare, axis=1)


    def display(self):
        # Create a VideoCapture object and read from input file
        # If the input is the camera, pass 0 instead of the video file name

        # Check if camera opened successfully
        camera = cv2.VideoCapture(0)
        if (camera.isOpened()== False): 
           print("Error opening video stream or file")

        # Read until video is completed
        while(camera.isOpened()):
          # Capture frame-by-frame
          ret, frame = camera.read()
          if ret == True:
            faces = self.trainer_obj.alignment.getAllFaceBoundingBoxes(frame)
            for face in faces:
                frame_aligned = self.trainer_obj.alignment.align(96, frame, face, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
                frame_aligned = (frame_aligned / 255.).astype(np.float32)
                pred_embedded = self.trainer_obj.model.predict(np.expand_dims(frame_aligned, axis=0))[0] 
                names = [name.split("/")[-2] for name in list(self.embedded.keys())]
                dist = self.face_distance(list(self.embedded.values()), pred_embedded)
                name = names[np.argmin(dist)] if any(x <= self.threshold for x in dist) else "unknown"
                ts = time.time()
                print(min(dist), name.upper(), datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))
                x1, y1, x2, y2 = face.left(), face.top() , face.left() + face.width(),face.top() + face.height()
                
                cv2.rectangle(frame,(x1, y1), (x2, y2), self.color, self.stroke)
                cv2.putText(frame, name, (x1,y1), self.font, 1, self.color, self.stroke, cv2.LINE_AA)


            frame = cv2.resize(frame, (1080, 920))  
            # Display the resulting frame
            cv2.imshow('Frame', frame)

            # Press <q> on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
              break

          # Break the loop
          else: 
            break

        # When everything done, release the video capture object
        camera.release()

        # Closes all the frames
        cv2.destroyAllWindows()
        for i in range (1,5):
            cv2.waitKey(1)


    def test(self):
        cap = cv2.VideoCapture(0)

        
        # Capture frame-by-frame
        ret, frame = cap.read()
        faces = self.trainer_obj.alignment.getAllFaceBoundingBoxes(frame)
        for face in faces:
            frame_aligned = self.trainer_obj.alignment.align(96, frame, face, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
            frame_aligned = (frame_aligned / 255.).astype(np.float32)
            pred_embedded = self.trainer_obj.model.predict(np.expand_dims(frame_aligned, axis=0))[0] 
            names = [name.split("/")[-2] for name in list(self.embedded.keys())]
            dist = self.face_distance(list(self.embedded.values()), pred_embedded)
            name = names[np.argmin(dist)] if any(x <= self.threshold for x in dist) else "unknown"
            ts = time.time()
            print(min(dist), name, datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))
            x1, y1, x2, y2 = face.left(), face.top() , face.left() + face.width(),face.top() + face.height()
            
            cv2.rectangle(frame,(x1, y1), (x2, y2), self.color, self.stroke)
            cv2.putText(frame, name, (x1,y1), self.font, 1, self.color, self.stroke, cv2.LINE_AA)


        frame = cv2.resize(frame, (1080, 920))  
        # Display the resulting frame
        cv2.imshow('frame',frame)   
    
        
        if cv2.waitKey(20) & 0xFF == ord('q'):
            # When everything done, release the capture
            cap.release()
            cv2.destroyAllWindows()

        
                
        
if __name__ == '__main__':
    static_img_obj = Videos()
    static_img_obj.display()




