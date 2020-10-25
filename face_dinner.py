import cv2
from retinaface.pre_trained_models import get_model
from deepface import DeepFace
from itertools import groupby
# import numpy as np

class face_dinner():
    def __init__(self):
        """ Get, Set, Initialize detectinator """
        self.model = get_model("resnet50_2020-07-20", max_size=2048)
        self.model.eval()
    
    def dine(self, image_path, factor=0.1):
        ''' Load image and infer bounding boxes for faces (MAX 5) '''
        image = cv2.imread(image_path)

        # annotations ==> [top_left_(x,y), bottom_right_(x,y)]
        annotations = self.model.predict_jsons(image)
        annotations = annotations[:5]

        faces = []
        # i = 1

        for ann in annotations:
            bb = ann['bbox']
            # 1 - y1
            # 3 - y2
            # 0 - x1
            # 2 - x2
            bb[1] = int(max(bb[1] - bb[1]*factor, 0))
            bb[3] = int(min(bb[3] + bb[3]*factor, image.shape[0]))
            bb[0] = int(max(bb[0] - bb[0]*factor, 0))
            bb[2] = int(min(bb[2] + bb[2]*factor, image.shape[1]))

            facex = image[bb[1]:bb[3], bb[0]:bb[2]]
            faces.append(facex)
            # cv2.imwrite(f'test{i}.jpg', facex)
            # i = i + 1

        fdata = self.deepfc(faces)
        zero, zero_str, zero_zero = [], [], [0]*4*5
        # rdata = self.recog(faces)

        for i in fdata:
            for _,j in enumerate(i):
                if _ == 1: zero_str.append(j)
                else: zero.append(j)

        zero_zero[:len(zero)] = zero # padding with zero
        zero_str = [word[0] for word in groupby(zero_str)] # removing duplicates

        return zero_zero, zero_str # [int_features, str_feature]


    def recog(self, cv_img):
        """ 
        Takes in opencv or numpy image and returns recognized names 
        in format ['John', 'Lafarge', '', 'Wick']
        for images (faces) of [john, lafarge, someone, wick]
        """
        pass

    def deepfc(self, cv_imgs):
        objs = DeepFace.analyze(cv_imgs, enforce_detection=False)

        race_labels = ['asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic']
        emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        gender_labels= ['Woman', 'Man']

        data = [] # [age, race, emotion, gender]
        for _ in range(len(cv_imgs)): # indexes increased by 1 to disambiguate from 0 (padded) ones
            obj = objs[f'instance_{_+1}']
            data.append([
                int(obj['age']), # age in int 
                obj["dominant_race"], # race string
                race_labels.index(obj["dominant_race"])+1, # non-zero int
                emotion_labels.index(obj["dominant_emotion"])+1, # non-zero int
                gender_labels.index(obj["gender"])+1 # non-zero int
            ])
            # print('img',_,'=>', int(obj["age"]),"years old",obj["dominant_race"],obj["dominant_emotion"],obj["gender"])

        return data


# if __name__ == '__main__':
#     fc = face_dinner()
#     print(fc.dine('images/img2.jpg'))

