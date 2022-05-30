## Related instructions
For the battery capacity set, the tasks are defined as: *No.5-No.6*, *No.6-No.5*, *No.5-No.7*, *No.7-No.5*, *No.6-No.7*, and *No.7-No.6*.

### The  functions  of each folder are described below

- ***Pre-train-model*** is used to store the pre-training model for each task named: ***taskname.pth***;
- ***data*** is used to store raw data;
- ***SeedIndex*** is used to record the index number of labeled samples in the target domain;
- ***Result*** is used to record the performance of DTRSR and other comparison models


### The functions of each ×× .py function are described below：

- ***model_define. py***  is use to train the pre-trained models saved in *Pre-trained-model* folder;
- ***MDA. py*** is used to measure the difference in marginal distribution between two datasets as follow:
- ***CDA. py*** is used to ,measure the difference in conditional distribution between two datasets;
- ***Result_evalute. py*** is used to evaluate the effect of regression, and an array composed of *MAE*,*MAPE*,*RMSE* and *R^2^* can be obtained;
- ***Seed_Module. py*** is used to perform a seed replacement module to integrate source domain knowledge and target domain knowledge into the new data in the form of **cluster structure** and **cluster center**, respectively;
- ***Model-contrast. py*** is used to compare *DTRSR* with other methods, such as *Pre_D*, *ReTrain*, *FC*, *MDA*, *CDA*, and *BDA*;