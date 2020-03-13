# Writer-Identification---Using-RNN

## Abstract:
writer identification is an important topic for pattern recognition. In this project, we idnetify the writers by using recurrent neural network
<br>
<br>

## Problem Description
* the handwriting data(500 Chinese character) was collected from each classmate 
* 300 character for training set, 100 character for validation set, 100 for testing(private, for TA to test the model) 
* Goal 1: 10-classes identification (10 classmate whose handwriting is far different from each other)
* Goal 2: 107-classes identification
<br>
<br>


## Result
#### 1-1 LSTM Network Parameters for 10-classes and 107-classes
<table> 
<tr><th> parameter of 10-classes </th><th> parameter of 107-classes </th></tr> 
<tr><td> 

| number of chosed RHS  | 3000 / 1000 |
|  -------------------- |----------------|
| Length of each RHS    | 50 / 100    |
| Feature Dimension     | 3              | 
| Model                 | LSTM Layer: 2 <br> hidden nodes: 300 <br> Bi-directional <br> batch size: 200 <br> batch_first: True |
| Epoch                 | 20             |

</td><td> 

| number of chosed RHS  | 3000 / 1000 |
|  -------------------- |----------------|
| Length of each RHS    | 50 / 100 / 150 |
| Feature Dimension     | 3              | 
| Model                 | LSTM Layer: 2 / 3 <br> hidden nodes: 256 / 500 / 800 <br> Bi-directional <br> batch size: 200 <br> batch_first: True |
| Epoch                 | 5             |


</td></tr> </table> 

#### 1-2 LSTM Result for 10-classes
| Length of Sequence | Training times(s) | Training Loss | Accuracy for each RHS (100%) | Accuracy for RHS(using Emsemble method)(100%) |
|----|----|----|----|----|
|50|722.69|0.1136|96.00|100.00|
|100|5647.02|0.1170|98.00|100.00|

#### 1-3 LSTM Result for 107-classes (different sequence)
| Length of Sequence | Training times(s) | Training Loss | Accuracy for each RHS (100%) | Accuracy for RHS(using Emsemble method)(100%) |
|----|----|----|----|----|
|50|7912.26|0.3047|86.00|100.00|
|100|8378.23|0.1542|91.00|100.00|
|150|8686.45|0.1560|93.00|100.00|

#### 1-4 LSTM Result for 107-classes (different hiddensize)
| Length of Sequence | Training times | Training Loss | Accuracy for each RHS (100%) | Accuracy for RHS(using Emsemble method)(100%) |
|----|----|----|----|----|
|256|8378.23|0.1542|91.00|100.00|
|500|12088.57|0.5424|91.00|100|
|800|13140.54|0.4854|92.00|100|
<br>

#### 2-1 GRU Network Parameters for 107-classes
| number of chosed RHS  | 3000 / 1000 |
|  -------------------- |----------------|
| Length of each RHS    | 50 / 100 / 150   |
| Feature Dimension     | 3              | 
| Model                 | GRU Layer: 1 <br> hidden nodes: 256 <br> batch size: 200 <br> batch_first: True |
| Epoch                 | 10             |

#### 2-2 GRU Result for 107-classes (different sequence)
| Length of Sequence | Training times(s) | Training Loss | Accuracy for each RHS | Accuracy for RHS(using Emsemble method) |
|----|----|----|----|----|
|50|163.380|1.995|0.39|102/107|
|100|353.14|2.449|0.33|98/107|
|150|502.114|1.280|0.57|105/107|
<br>
<br>

## Analysis
* Training time: GRU < LSTM, because:
   * GRU (two gate), LSTM (three gate)
   * GRU has no bi-directional
* Performance: LSTM is better, because:
   * LSTM model is more complicate, and can learn more important feature

### more details -> in the pdf file: 



## Run the Codes
#### 1 Codes/
* Codes folder contain two model: LSTM, GRU
##### How to run 
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
    
#### 2 TestCodes/
* put the validation-set(`Validation_with_labels`)into TestCodes folder
* run `test.py` by `python test.py --testfolder ../Validation_with_labels --num_class 10` where `../Validation_with_labels` is the path of validation set, 10/107 is number of classes
    

