# Tutorial for development system

## For processed data, please refer the 'RRT_processed_data.zip' files

## 1. Install requirement package base on the tutorial in 'requirement.txt'

## 2. Training
- Refer 'Detection_train.ipynb'.

- Change the directory based on your specific data directory.

- Train process includes:

    (a) Window Interval Processing (ts_dataloader) (refer 'dataloader.py')
  
        +  For 'ts_dataloader' function, you can select your specific input features ('lab_list', 'sign_list', 'dem_list').
  
        + You can also select your specific task corresponding each label column.(Tasks colum's name includes: 'label' (normal or abnormal), 'is_detection', 'is_event', 'ev_w_dec').
  
        + You can also select 'window_len' (history data using for prediction); 'stride' (Number of future timesteps we want to predict).
  
        + Finally, we have sequences process data for training DL models.
  
        Specifically for SiameseTS_model, It should includes 2 type of input features: 'x_t' (time based features); 'x_d' (non-time based features).

    (b) Training
  
        + Change your specific training folder to save the trained models.
  
        + Select model (refer 'model.py')
  
        Example: 'model = SiamseTS_model(x_t.shape[1:], x_d.shape[1:], num_classes=num_classes)'

    You can design your training strategy to improve the model's performance.

## 3. Evaluation
- Refer 'Detection_evaluation.ipynb'.

- Change the directory based on your specific data directory.

- Using ts_dataloader for test data. Input features list and task should be similar with the trained model.

- Get prediction and evaluation (refer 'eval.py')

