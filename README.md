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
 Epoch 50/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 3s 3s/step - accuracy: 0.4062 - loss: 1.4615 - val_accuracy: 0.6667 - val_loss: 1.3491

VGG16:
Final accuracy: 0.3333

Classification Report:
                        precision    recall  f1-score   support

            Foam-Heavy       0.38      1.00      0.55         3
             Foam-mild       0.00      0.00      0.00         1
Post-Antifoam Addition       0.00      0.00      0.00         1
           Foam-Medium       0.00      0.00      0.00         1
               No Foam       1.00      0.33      0.50         3

              accuracy                           0.44         9
             macro avg       0.28      0.27      0.21         9
          weighted avg       0.46      0.44      0.35         9


ResNet50:
Final accuracy: 0.3333

Classification Report:
                        precision    recall  f1-score   support

            Foam-Heavy       0.00      0.00      0.00         3
             Foam-mild       0.00      0.00      0.00         1
Post-Antifoam Addition       0.00      0.00      0.00         1
           Foam-Medium       0.00      0.00      0.00         1
               No Foam       0.33      1.00      0.50         3

              accuracy                           0.33         9
             macro avg       0.07      0.20      0.10         9
          weighted avg       0.11      0.33      0.17         9


MobileNetV2:
Final accuracy: 0.6667

Classification Report:
                        precision    recall  f1-score   support

            Foam-Heavy       0.67      0.67      0.67         3
             Foam-mild       0.00      0.00      0.00         1
Post-Antifoam Addition       0.00      0.00      0.00         1
           Foam-Medium       0.00      0.00      0.00         1
               No Foam       0.33      0.67      0.44         3

              accuracy                           0.44         9
             macro avg       0.20      0.27      0.22         9
          weighted avg       0.33      0.44      0.37         9
```
