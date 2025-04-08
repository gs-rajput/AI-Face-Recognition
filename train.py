import cv2
import os
import face_recognition
import pickle

dataset_dir = "dataset"
encodings_dir = "encodings"

os.makedirs(encodings_dir, exist_ok=True)

name = input("Enter your name: ").strip()
person_dir = os.path.join(dataset_dir, name)
os.makedirs(person_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0
max_images = 100  # Changed from 5 to 20

print(f"[INFO] Press 's' to start capturing {max_images} images for {name}...")

# Wait until user presses 's' to start
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow("Press 's' to start", frame)
    key = cv2.waitKey(1)
    if key == ord('s'):
        break
    elif key == 27:  # ESC to quit
        cap.release()
        cv2.destroyAllWindows()
        exit()

print(f"[INFO] Starting capture...")

while count < max_images:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)

    if boxes:
        count += 1
        img_path = os.path.join(person_dir, f"{count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"[INFO] Captured {count}/{max_images}")

    cv2.imshow("Capture", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Encode and save
print("[INFO] Encoding faces...")
known_encodings = []
known_names = []

for img_file in os.listdir(person_dir):
    img_path = os.path.join(person_dir, img_file)
    image = face_recognition.load_image_file(img_path)
    boxes = face_recognition.face_locations(image)
    encodings = face_recognition.face_encodings(image, boxes)

    for encoding in encodings:
        known_encodings.append(encoding)
        known_names.append(name)

with open(os.path.join(encodings_dir, f"{name}.pkl"), "wb") as f:
    pickle.dump((known_names, known_encodings), f)

print("[INFO] Training complete and saved.")