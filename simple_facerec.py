import face_recognition
import cv2
import os
import glob
import numpy as np

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

        # Resize frame for a faster speed
        self.frame_resizing = 0.25
        
        # this code for file images have more not same images [ one folder]
    # def load_encoding_images(self, images_path):
    #     images_path = glob.glob(os.path.join(images_path, "*.*"))
    #     print("{} encoding images found.".format(len(images_path)))

    #     for img_path in images_path:
    #         try:
    #             img = cv2.imread(img_path)
    #             if img is None:
    #                 print(f"Error reading image: {img_path}")
    #                 continue

    #             rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #             basename = os.path.basename(img_path)
    #             (filename, ext) = os.path.splitext(basename)
    #             img_encoding = face_recognition.face_encodings(rgb_img)[0]

    #             self.known_face_encodings.append(img_encoding)
    #             self.known_face_names.append(filename)
    #         except Exception as e:
    #             print(f"Error processing image: {img_path}. Error: {e}")

    #     print("Encoding images loaded")
        


        # this code to read more images from 6 folders, where each folder contains images of one person [more folder]
    def load_encoding_images(self, base_folder):
        for person_folder in os.listdir(base_folder):
            person_path = os.path.join(base_folder, person_folder)
            if os.path.isdir(person_path):
                person_images = glob.glob(os.path.join(person_path, "*.*"))
                
                if not person_images:
                    print(f"No images found for {person_folder}. Skipping...")
                    continue

                print(f"Loading images for {person_folder}...")
                
                person_encodings = []
                for img_path in person_images:
                    try:
                        img = cv2.imread(img_path)
                        if img is None:
                            print(f"Error reading image: {img_path}")
                            continue

                        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img_encoding = face_recognition.face_encodings(rgb_img)[0]

                        person_encodings.append(img_encoding)
                    except Exception as e:
                        print(f"Error processing image: {img_path}. Error: {e}")

                if person_encodings:
                    # Use the average encoding for the person
                    avg_encoding = np.mean(person_encodings, axis=0)
                    self.known_face_encodings.append(avg_encoding)
                    self.known_face_names.append(person_folder)

        print("Encoding images loaded")


    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        # Find all the faces and face encodings in the current frame of video
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        # Convert to numpy array to adjust coordinates with frame resizing quickly
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names
