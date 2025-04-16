## Multi-View  Fused Nonnegative Matrix Completion Methods for  Drug-Target Interaction Prediction

A Python implementation of MvFNMC methods. 
MvFNMC is a computational pipeline to predict novel drug-target interactions (DTIs) from heterogeneous network. 

## Requirement
joblib==0.14.1    
numpy==1.17.4    
scikit-learn==1.0.1    
scikit-multilearn==0.2.0     
scipy==1.6.1    
The codes were tested on Python 3.7




To get the results of different methods, please run PyDTI.py by setting suitable values for the following parameters:

- --method:  			set DTI prediction method    
- --dataset: 			choose the benchmark dataset, i.e., nr, gpcr, ic, e
- --csv:				choose the cross-validation setting, 1 for CVS1, 2 for CVS2, and 3 for CVS3, (default 1)



## Quick start
- Run `MvFNMC/PyDTI.py --method="mvfnmc1new1" --dataset="nr"`: predict drug-target interactions, and evaluate the results with cross-validation for the four golden standard datasets: Enzymes (Es), Ion Channels (IC), G Protein-Coupled Receptors (GPCRs), and Nuclear Receptors (NRs).
- Run `MvFNMC_luo/PyDTI.py --method="mvfnmc1new1" --dataset="luo"`: predict drug-target interactions, and evaluate the results with cross-validation for the Luo dataset.


### Data

- the folder `MvFNMC/data-multiviews` includes all the data of Es, IC, GPCRs, NRs
- the folder `MvFNMC_luo/luo` includes all the data of the Luo dataset.


### Contacts
If you have any questions or comments, please feel free to email Ting Li (litingmath@163.com).
