# reactorMonitoring
 
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
Final accuracy: 0.9540

Classification Report:
                        precision    recall  f1-score   support

            Foam-Heavy       0.96      0.99      0.98       100
             Foam-mild       0.87      0.95      0.91       100
Post-Antifoam Addition       1.00      0.97      0.98       100
           Foam-Medium       0.97      0.90      0.93       100
               No Foam       0.98      0.96      0.97       100

              accuracy                           0.95       500
             macro avg       0.96      0.95      0.95       500
          weighted avg       0.96      0.95      0.95       500
```
