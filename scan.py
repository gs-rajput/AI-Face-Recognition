# import cv2
# import os
# import face_recognition
# import pickle
# import numpy as np
#
# encodings_dir = "encodings"
#
# known_encodings = []
# known_names = []
#
# for file in os.listdir(encodings_dir):
#     with open(os.path.join(encodings_dir, file), "rb") as f:
#         names, encs = pickle.load(f)
#         known_names.extend(names)
#         known_encodings.extend(encs)
#
# cap = cv2.VideoCapture(0)
# print("[INFO] Scanning... Press ESC to exit.")
#
# while True:
#     ret, frame = cap.read()
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     boxes = face_recognition.face_locations(rgb)
#     encodings = face_recognition.face_encodings(rgb, boxes)
#
#     for (top, right, bottom, left), encoding in zip(boxes, encodings):
#         matches = face_recognition.compare_faces(known_encodings, encoding)
#         name = "Unknown"
#
#         if True in matches:
#             matched_idxs = [i for i, b in enumerate(matches) if b]
#             counts = {}
#             for i in matched_idxs:
#                 counts[known_names[i]] = counts.get(known_names[i], 0) + 1
#             name = max(counts, key=counts.get)
#
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
#         cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
#
#     cv2.imshow("Scan", frame)
#     if cv2.waitKey(1) == 27:
#         break
#
# cap.release()
# cv2.destroyAllWindows()
import cv2
import os
import face_recognition
import pickle
import numpy as np

encodings_dir = "encodings"

known_encodings = []
known_names = []

# Load all encodings
for file in os.listdir(encodings_dir):
    with open(os.path.join(encodings_dir, file), "rb") as f:
        names, encs = pickle.load(f)
        known_names.extend(names)
        known_encodings.extend(encs)

cap = cv2.VideoCapture(0)
print("[INFO] Scanning... Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, boxes)

    for (top, right, bottom, left), encoding in zip(boxes, encodings):
        face_distances = face_recognition.face_distance(known_encodings, encoding)
        if len(face_distances) == 0:
            continue

        best_match_index = np.argmin(face_distances)
        confidence = (1 - face_distances[best_match_index]) * 100

        if confidence > 60:  # threshold for better reliability
            name = known_names[best_match_index]
            label = f"{name} ({confidence:.2f}%)"
            print(f"[INFO] Match found: {name} (Confidence: {confidence:.2f}%)")
        else:
            name = "Unknown"
            label = name
            print("[INFO] Face detected but not recognized")

        # Draw on frame
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow("Scan", frame)
    if cv2.waitKey(1) == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
