import os
import cv2
import torch
import pickle
import numpy as np
import time
from facenet_pytorch import InceptionResnetV1, MTCNN

os.makedirs("dl_encodings", exist_ok=True)
os.makedirs("dl_dataset", exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(keep_all=False, device=device)

name = input("Enter your name: ").strip()
dataset_dir = os.path.join("dl_dataset", name)
os.makedirs(dataset_dir, exist_ok=True)

count = 0
max_images = 100
print(f"[INFO] Press 's' to start capturing {max_images} images...")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    cv2.imshow("Capture", frame)
    key = cv2.waitKey(1)
    if key == ord('s'):
        print(f"[INFO] Starting capture for {name}...")
        while count < max_images:
            ret, frame = cap.read()
            if not ret:
                continue
            img_path = os.path.join(dataset_dir, f"{count+1}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"[INFO] Captured {count+1}/{max_images}")
            count += 1
            time.sleep(0.1)
        break
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("[INFO] Encoding faces with deep learning model...")
embeddings = []
for img_name in os.listdir(dataset_dir):
    img_path = os.path.join(dataset_dir, img_name)
    img = cv2.imread(img_path)
    face = mtcnn(img)
    if face is not None:
        face_embedding = resnet(face.unsqueeze(0).to(device)).detach().cpu().numpy()[0]
        embeddings.append(face_embedding)

if embeddings:
    with open(f"dl_encodings/{name}.pkl", "wb") as f:
        pickle.dump((name, embeddings), f)
    print("[INFO] Saved deep embeddings.")
else:
    print("[WARN] No faces found to encode.")

