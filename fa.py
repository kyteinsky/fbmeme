import face_alignment
# from skimage import io

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')

preds = fa.get_landmarks_from_directory('images/')

