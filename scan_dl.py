import os
import cv2
import torch
import pickle
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.metrics.pairwise import cosine_similarity

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(keep_all=True, device=device)  # Enable multi-face detection

# Load Encodings
encodings_dir = "dl_encodings"
known_embeddings = []
known_names = []

for file in os.listdir(encodings_dir):
    with open(os.path.join(encodings_dir, file), "rb") as f:
        name, embs = pickle.load(f)
        known_names.extend([name] * len(embs))
        known_embeddings.extend(embs)

known_embeddings = np.array(known_embeddings)

# Scan
cap = cv2.VideoCapture(0)
print("[INFO] Scanning... Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    boxes, probs = mtcnn.detect(frame)
    faces = mtcnn(frame)

    if faces is not None and boxes is not None:
        for face, box in zip(faces, boxes):
            if face is None or box is None:
                continue

            emb = resnet(face.unsqueeze(0).to(device)).detach().cpu().numpy()
            sims = cosine_similarity(emb, known_embeddings)[0]
            best_idx = np.argmax(sims)
            confidence = sims[best_idx] * 100
            name = known_names[best_idx] if confidence > 60 else "Unknown"

            # Draw box & label
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({confidence:.2f}%)", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow("Scan DL", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
