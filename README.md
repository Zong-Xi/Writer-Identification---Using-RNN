# Writer-Identification---Using-RNN

## Abstract:
writer identification is an important topic for pattern recognition. In this project, we idnetify the writers by using recurrent neural network

## Problem Description
* the handwriting data(500 Chinese character) was collected from each classmate 
* 300 character for training set, 100 character for validation set, 100 for testing(private, for TA to test the model) 
* Goal 1: 10-classes identification (10 classmate whose handwriting is far different from each other)
* Goal 2: 107-classes identification

## 






## Run the Codes
### 1. Codes/
* Codes folder contain two model: LSTM, GRU
### How to run 
1. put the dataset in the data-folder
2. setup the path of training-data in the `main` function -> `path_train`
3. setup the path of testing-data in the `main` function -> `path_test`
4. the `main` function :
    * LSTM on 10-classes: lstm/main_lstm_10.py
    * LSTM on 107-classes: lstm/main_lstm_107.py
    * GRU on 10-classes: gru/main_gur_10.py
    * GRU on 107-classes: gru/main_gru_107.py
5. the output:
    * loss for each epoch (save automatically)
    * training time
    * the accuracy for one RHS
    * the accuracy for RHS(using Emsemble method), and save the model in the model-folder
    
### 2. TestCodes/
* put the validation-set(`Validation_with_labels`)into TestCodes folder
* run `test.py` by `python test.py --testfolder ../Validation_with_labels --num_class 10` where `../Validation_with_labels` is the path of validation set, 10/107 is number of classes
    

