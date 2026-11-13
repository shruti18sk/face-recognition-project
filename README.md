# Face Recognition with Embeddings

A face recognition pipeline built with the `face_recognition` library (dlib, 128-dimensional face embeddings). It enrolls a few known people from a single photo each, then identifies faces in new photos or marks them as unknown. The notebook also visualises how the embeddings behave.

## What it does
1. Detects faces and draws bounding boxes
2. Converts each face into a 128-number embedding
3. Matches a test face to the closest enrolled face using Euclidean distance
4. Marks the face as `unknown` if the distance is above a threshold
5. Visualises embeddings with t-SNE and compares same-person vs different-person distances
6. Evaluates accuracy at different thresholds and shows the mistakes

## Setup
- Dataset: subset of LFW (Labeled Faces in the Wild), 7 people, 1288 images
- Enrolled (known) people: 5, with 1 photo each
- Test set: 140 photos (100 of known people, 40 of 2 people who were never enrolled)
- Images resized to 500 px width before encoding

## Results

| Threshold | Accuracy | False accepts | False rejects | Wrong person | No face found |
|---|---|---|---|---|---|
| 0.6 | 138/140 = 98.6% | 1 | 1 | 0 | 0 |
| 0.5 | 134/140 = 95.7% | 0 | 6 | 0 | 0 |

A stricter threshold removed false accepts but caused more false rejects.

![Results](results.png)

## Mistakes at threshold 0.6
- One photo contained several faces and the system used a different face than the intended one, so the person was not recognised.
- One unknown person was matched to a known person at a borderline distance of 0.60.

## Limitations
- Small evaluation: only 5 enrolled people and 2 unknown people
- LFW photos are mostly frontal, so real-world accuracy would likely be lower
- No liveness detection or anti-spoofing yet, so a printed photo could fool it

## Future work
- Real-time recognition from a webcam
- Liveness detection / anti-spoofing
- Larger and more varied evaluation set
- Pick the largest face when a photo contains several faces

## Tech stack
Python, face_recognition (dlib), OpenCV, NumPy, scikit-learn (t-SNE), Matplotlib, Google Colab

## How to run
Open the notebook in Google Colab and run the cells from top to bottom. The first cell installs `face_recognition`, and the dataset is downloaded automatically.
