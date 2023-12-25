import face_recognition
import cv2
from datetime import datetime
import csv
import numpy as np
import os 


folder = 'images'
images = []
for file in os.listdir(folder):
    if file.endswith('.jpg') or file.endswith('.png'):
        images.append(file)
video_capture = cv2.VideoCapture(0)       
# if you use mobile webcam 
# video_capture = cv2.VideoCapture("http://192.168.43.1:8080/video")
# width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

face_encodings_list = []

for a in images:
    load=face_recognition.load_image_file('images/'+a)
    a_encoding=face_recognition.face_encodings(load)[0]
    face_encodings_list.append(a_encoding)

print(face_encodings_list)

know_face_name = []
for i, s in enumerate(images):
    
    words = s.split()
    word1=str(words)
    word1=word1[2:-6]
    know_face_name.append(word1)

student=know_face_name.copy()

face_location=[]
face_encoding=[]
face_name=[]
process_this_frame = True

now=datetime.now()
current_date=now.strftime("%Y-%m-%d")


csv_data = []
while True:
    
    
    _,frame= video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]

   
    if process_this_frame:
       
       
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        
        for face_encoding in face_encodings:
            
            matches = face_recognition.compare_faces(face_encodings_list,face_encoding,)
            name = "Unknown"

            face_distances = face_recognition.face_distance(face_encodings_list, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = know_face_name[best_match_index]
            
            face_names.append(name)

            if name in know_face_name:
                if name in student:
                    student.remove(name)
                    print(student)
                    current_time = now.strftime("%H-%M-%S %p")
                    csv_data.append(name)
                    csv_data.append(current_time)
                    with open(current_date+'.csv', 'a', newline='') as csv_file:
                        csv_writer = csv.writer(csv_file)
                        csv_writer.writerow(csv_data)
                        print(csv_data)
                    csv_data.clear()
            if len(student)==0:
                csv_file.close()
                exit()

    
                
       
    process_this_frame = not process_this_frame
    
    for (top, right, bottom, left), name in zip(face_locations, face_names):
   
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

       
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    
    cv2.imshow('Attendance Systeam', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()