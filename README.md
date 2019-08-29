# SelfDriving_Action_Prediction
Multiple ego-actions and explanations predictions on new annotated self-driving dataset

## Baseline
- Architecture: ResNet101 with linear classifier
- Usage:
**1. Predicting multiple ego-actions:**
```python baseline/train_cnn.py --image_root path/to/image/ --gtroot path/to/groundtruth/actions/ --out_dir output/directory/```
**2. Predicting multiple ego-actions and explanations**
```python baseline/train_cnn.py --image_root path/to/image/ --gtroot path/to/groundtruth/actions/ --out_dir output/directory/ --side```


## Object-centric model
- Architecture: FasterRCNN with selector and predictor
