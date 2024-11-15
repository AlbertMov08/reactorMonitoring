# reactorMonitoring

## End Goal Implementation
```
webcam hooked up with computer, computer talking with server for ai software, evaluating images every five minutes, find sampling interval that makes sense
evaluating across 2-4 different vessels
if foaming event is seen in any vessel, can send signal to controller, dose in antifoam
send notification to operator about problem
```
 
## CNN Implementation
```
Number of classes: 5
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                      │ (None, 222, 222, 32)        │             896 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d (MaxPooling2D)         │ (None, 111, 111, 32)        │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_1 (Conv2D)                    │ (None, 109, 109, 64)        │          18,496 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_1 (MaxPooling2D)       │ (None, 54, 54, 64)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_2 (Conv2D)                    │ (None, 52, 52, 64)          │          36,928 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten (Flatten)                    │ (None, 173056)              │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 64)                  │      11,075,648 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 6)                   │             390 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 11,132,358 (42.47 MB)
 Trainable params: 11,132,358 (42.47 MB)
 Non-trainable params: 0 (0.00 B)

Custom CNN:
Final accuracy: 0.7480

Classification Report:
                        precision    recall  f1-score   support

            Foam-Heavy       0.78      0.70      0.74       100
             Foam-mild       0.66      0.64      0.65       100
Post-Antifoam Addition       0.86      0.85      0.85       100
           Foam-Medium       0.65      0.69      0.67       100
               No Foam       0.80      0.86      0.83       100

              accuracy                           0.75       500
             macro avg       0.75      0.75      0.75       500
          weighted avg       0.75      0.75      0.75       500


VGG16:
Final accuracy: 0.8460

Classification Report:
                        precision    recall  f1-score   support

            Foam-Heavy       0.94      0.79      0.86       100
             Foam-mild       0.85      0.71      0.77       100
Post-Antifoam Addition       0.82      0.97      0.89       100
           Foam-Medium       0.82      0.86      0.84       100
               No Foam       0.83      0.90      0.86       100

              accuracy                           0.85       500
             macro avg       0.85      0.85      0.84       500
          weighted avg       0.85      0.85      0.84       500


ResNet50:
Final accuracy: 0.4700

Classification Report:
                        precision    recall  f1-score   support

            Foam-Heavy       0.29      0.16      0.21       100
             Foam-mild       0.42      0.40      0.41       100
Post-Antifoam Addition       0.55      0.96      0.70       100
           Foam-Medium       0.38      0.30      0.33       100
               No Foam       0.57      0.53      0.55       100

              accuracy                           0.47       500
             macro avg       0.44      0.47      0.44       500
          weighted avg       0.44      0.47      0.44       500


MobileNetV2:
Final accuracy: 0.9700

Classification Report:
                        precision    recall  f1-score   support

            Foam-Heavy       1.00      1.00      1.00        40
             Foam-mild       0.97      0.93      0.95        40
Post-Antifoam Addition       1.00      1.00      1.00        40
           Foam-Medium       0.95      0.95      0.95        40
               No Foam       0.93      0.97      0.95        40

              accuracy                           0.97       200
             macro avg       0.97      0.97      0.97       200
          weighted avg       0.97      0.97      0.97       200

ZeroShot OpenAI Model:
Accuracy: 0.3265

Classification Report:
                        precision    recall  f1-score   support

            Foam-Heavy       0.59      0.63      0.61        30
             Foam-Mild       0.30      0.44      0.36        34
Post-Antifoam Addition       0.00      0.00      0.00        22
           Foam-Medium       0.17      0.38      0.24        21
               No Foam       0.60      0.15      0.24        40
                 Error       0.00      0.00      0.00         0

              accuracy                           0.33       147
             macro avg       0.28      0.27      0.24       147
          weighted avg       0.38      0.33      0.31       147

Ensemble Model Results:
Best weights found - MobileNetV2: 0.30, ViT: 0.70
Accuracy: 0.9750
Classification Report:
                        precision    recall  f1-score   support

            Foam-Heavy       1.00      1.00      1.00        40
             Foam-mild       0.97      0.95      0.96        40
Post-Antifoam Addition       1.00      1.00      1.00        40
           Foam-Medium       0.97      0.95      0.96        40
               No Foam       0.93      0.97      0.95        40

              accuracy                           0.97       200
             macro avg       0.98      0.97      0.98       200
          weighted avg       0.98      0.97      0.98       200
```
