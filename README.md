# Face Recognition with Embeddings

A face recognition pipeline built with the `face_recognition` library (dlib, 128-dimensional face embeddings). It enrolls known people from a single photo each, then identifies faces in new photos or marks them as unknown. The notebook also visualises how the embeddings behave and studies how the matching threshold changes the results.

## What it does
1. Detects faces and draws bounding boxes (the largest face is used when a photo has several)
2. Converts each face into a 128-number embedding
3. Matches a test face to the closest enrolled face using Euclidean distance
4. Marks the face as `unknown` if the distance is above a threshold
5. Visualises embeddings with t-SNE and compares same-person vs different-person distances
6. Evaluates accuracy at different thresholds and shows the mistakes
7. Repeats the evaluation on a larger test with many more unknown people

## Small test (7 people)
- Dataset: subset of LFW (Labeled Faces in the Wild), 7 people, 1288 images
- Enrolled (known) people: 5, with 1 photo each
- Test set: 140 photos (100 of known people, 40 of 2 people who were never enrolled)
- Images resized to 500 px width before encoding

| Threshold | Accuracy | False accepts | False rejects | Wrong person | No face found |
|---|---|---|---|---|---|
| 0.6 | 139/140 = 99.3% | 1 | 0 | 0 | 0 |
| 0.5 | 135/140 = 96.4% | 0 | 5 | 0 | 0 |

![Matching results on the small test](results.png)

### Mistakes at threshold 0.6
- An earlier version missed one person because the photo contained several faces and the system picked the wrong one. The pipeline now uses the largest detected face, which fixed this (7 of the 140 test photos contain more than one face).
- One unknown person is still matched to a known person at a borderline distance of 0.60.

## Larger test (62 people)
30 people enrolled with 1 photo each; 620 test images, including 320 photos of 32 people who were never enrolled.

| Threshold | Accuracy | False accepts | False rejects | Wrong person |
|---|---|---|---|---|
| 0.6 (default) | 564/620 = 91.0% | 53 | 1 | 2 |
| 0.56 (chosen on this test set) | 603/620 = 97.3% | 8 | 7 | 2 |
| 0.5 | 559/620 = 90.2% | 0 | 61 | 0 |

The two error rates are closest at a threshold of about 0.56 (false accept 2.5%, false reject 2.3%).

**Main finding:** the default threshold of 0.6 looked fine on the small test, but with 32 unknown people it let many strangers through (53 of 320 unknown photos). A threshold that works on a small test does not necessarily carry over to a larger one.

## Limitations
- The 0.56 threshold was picked on the same test set, so its accuracy is likely slightly optimistic
- LFW photos are mostly frontal, so real-world accuracy would likely be lower
- No liveness detection or anti-spoofing yet, so a printed photo could fool it
- The embedding model is pre-trained; it was not trained or fine-tuned here

## Future work
- Choose the threshold on one set of people and evaluate on a separate set
- Test on larger and more varied data (non-frontal faces, different cameras and lighting)
- Real-time recognition from a webcam
- Liveness detection / anti-spoofing
- Compare with other embedding models

## Tech stack
Python, face_recognition (dlib), OpenCV, NumPy, scikit-learn (t-SNE), Matplotlib, Google Colab

## How to run
Open the notebook in Google Colab and run the cells from top to bottom. The first cell installs `face_recognition`, and the dataset is downloaded automatically. The larger test at the end takes a few minutes to run.
