# Writer-Recognition---Using-RNN

### 1. Codes folder:
* Codes folder contain two model: LSTM, GRU
### How to run 
1. put the dataset in the data-folder
2. setup the path of training-data in the `main` function -> `path_train`
3. setup the path of testing-data in the `main` function -> `path_test`
4. the `main` function :
    * LSTM on 10-classification: lstm/main_lstm_10.py
    * LSTM on 107-classification: lstm/main_lstm_107.py
    * GRU on 10-classification: gru/main_gur_10.py
    * GRU on 107-classification: gru/main_gru_107.py
5. the output:
    
运行输出以下内容： 
（1）每次迭代的loss（自动保存成） 
（2）训练时间training time 
（3）测试集中单个RHS准确率 
（4）测试集中投票结果RHS准确率 并且自动保存模型在model文件夹里，可供test测试

2.TestCodes/ #测试代码，含test.py 
将待运行的验证集Validation_with_labels放到TestCodes文件夹底下 
使用命令''python test.py --testfolder ../Validation_with_labels（测试目录） --num_class 10（或107，分类类别数）''运行test.py；
