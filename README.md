# SelfDriving_Action_Prediction
Multiple ego-actions and explanations predictions on new annotated self-driving dataset

## Baseline
- Architecture: ResNet101 with linear classifier  
- Usage:  
**1. Predicting multiple ego-actions:**  
Training:  
```python baseline/train_cnn.py --imageroot path/to/image/ --gtroot path/to/groundtruth/actions/ --out_dir output/directory/```  
Testing:
```python baseline/test_cnn.py --imageroot path/to/image/ --gtroot path/to/groundtruth/actions/ --model_root path/to/trained/model/weights```  

**2. Predicting multiple ego-actions and explanations**  
Training:  
```python baseline/train_cnn.py --imageroot path/to/image/ --gtroot path/to/groundtruth/actions/ --resonroot path/to/groundtruth/explanations --out_dir output/directory/ --side```  
Testing:
```python baseline/test_cnn.py --imageroot path/to/image/ --gtroot path/to/groundtruth/actions/ --resonroot path/to/groundtruth/explanations --model_root path/to/trained/model/weights --side```  


## Object-centric model  
- Architecture: FasterRCNN with selector and predictor  
```cd maskrcnn-benchmark/```  
**1. Preparation:**  
Follow [INSTALL.md](INSTALL.md) for installation instructions.  
**2. Pretrained faster-rcnn**  
Follow [README.md](README.md) for training and testing faster-rcnn.  
Using "e2e_faster_rcnn_R_50_C4_1x.yaml" as config file.
Training:  
```python action_prediction/train_all.py --imageroot path/to/image/ --gtroot path/to/groundtruth/actions/ --resonroot path/to/groundtruth/explanations --model_root path/to/pretrained/fasterrcnn/weights OUT_DIR output/directory MODEL.SIDE True```  
Testing:  
```python action_prediction/test_all.py --imageroot path/to/image/ --gtroot path/to/groundtruth/actions/ --resonroot path/to/groundtruth/explanations --model_root path/to/pretrained/fasterrcnn/weights OUT_DIR output/directory MODEL.SIDE True```  

( Optional paramters:  
  --initLR  
  --weight_decay  
  --num_epoch  
  --batch_size  
)  
