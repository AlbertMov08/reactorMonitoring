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
Final accuracy: 0.6364

Classification Report:
                        precision    recall  f1-score   support

            Foam-Heavy       0.33      0.20      0.25         5
             Foam-mild       0.60      0.75      0.67         4
Post-Antifoam Addition       0.60      0.75      0.67         4
           Foam-Medium       0.75      0.60      0.67         5
               No Foam       0.80      1.00      0.89         4

              accuracy                           0.64        22
             macro avg       0.62      0.66      0.63        22
          weighted avg       0.61      0.64      0.61        22


VGG16:
Final accuracy: 0.5000

Classification Report:
                        precision    recall  f1-score   support

            Foam-Heavy       0.33      0.20      0.25         5
             Foam-mild       0.43      0.75      0.55         4
Post-Antifoam Addition       0.60      0.75      0.67         4
           Foam-Medium       0.33      0.20      0.25         5
               No Foam       0.75      0.75      0.75         4

              accuracy                           0.50        22
             macro avg       0.49      0.53      0.49        22
          weighted avg       0.47      0.50      0.47        22


ResNet50:
Final accuracy: 0.4091

Classification Report:
                        precision    recall  f1-score   support

            Foam-Heavy       0.25      0.20      0.22         5
             Foam-mild       0.00      0.00      0.00         4
Post-Antifoam Addition       0.43      0.75      0.55         4
           Foam-Medium       0.50      0.20      0.29         5
               No Foam       0.57      1.00      0.73         4

              accuracy                           0.41        22
             macro avg       0.35      0.43      0.36        22
          weighted avg       0.35      0.41      0.35        22


MobileNetV2:
Final accuracy: 0.6364

Classification Report:
                        precision    recall  f1-score   support

            Foam-Heavy       1.00      0.40      0.57         5
             Foam-mild       0.25      0.25      0.25         4
Post-Antifoam Addition       0.67      1.00      0.80         4
           Foam-Medium       0.67      0.80      0.73         5
               No Foam       0.75      0.75      0.75         4

              accuracy                           0.64        22
             macro avg       0.67      0.64      0.62        22
          weighted avg       0.68      0.64      0.62        22
```
