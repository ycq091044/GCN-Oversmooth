# Revisiting Over-smoothing in Deep GCNs
Here are the complete codes and datasets that I used for "Revisiting Over-smoothing in Deep GCNs".
They are categorized into different folders according to the experimental sections in the paper. Please contact 
 me <chaoqiy2@illinois.edu> if you have any question.

## Reproductive code folder structure
- **data/**
    - This is the data folder for Cora, Citeseer, Pubmed

- **Karate_Demo.ipynb**
    - Here is the data, model and vis code for Figure 1

- **Karate_Demo2.ipynb**
    - Here is the data, model and vis code for Figure 2

- **mean-subtraction/ (reproductive codes for Experiment Section 5.2 Mean-subtraction for GCNs)**
    - Instructions
        - Before training, please use ```mkdir cora``` or ```mkdir citeseer``` or ```mkdir pubmed``` to generate the result folders for Cora, Citeseer, Pubmed
        - Run ```python train2.py``` or ```python train.py``` and get the experimental results for 20 rounds
        - move three result folders to Result-and-Vis
    - Result-and-Vis/
        - We already have the results for this paper (Cora, Citeseer, Pubmed). Users could choose to reproduce it by running ```train2.py``` or ```train.py```
        - run ```mean-subtraction-vis.ipynb``` to generate the figures

- **neighbor-aggregation-weight/ (reproductive codes for Experiment Section 5.3 Weight of Neighborhood Aggregation in GCNs)**
    - Instructions
        - Before training, please use ```mkdir cora``` or ```mkdir citeseer``` or ```mkdir pubmed``` to generate the result folders for Cora, Citeseer, Pubmed
        - Run ```python train.py --dataset [name of the dataset]``` and get the experimental results for 20 rounds

- **performace-depth-oversmooth/ (reproductive codes for Experiment Section 5.1 Overfitting in Deep GCNs Part-I)**
    - Instructions
        - Before training, please use ```mkdir cora``` or ```mkdir citeseer``` or ```mkdir pubmed``` to generate the result folders for Cora, Citeseer, Pubmed        
    	- Run ```python train.py``` and get the experimental results for 20 rounds
        - move three result folders to Result-and-Vis
    - Result-and-Vis/
        - We already have the results for this paper (Cora, Citeseer, Pubmed). Users could choose to reproduce it by running train.py
        - run ```three-set-running-vis.ipynb``` to generate the figures

- **performace-depth2-loss-function/ (reproductive codes for Experiment Section 5.1 Overfitting in Deep GCNs Part-II)**
    - Instructions
        - Before training, please use ```mkdir cora``` or ```mkdir citeseer``` or ```mkdir pubmed``` to generate the result folders for Cora, Citeseer, Pubmed  
        - Run ```python train.py``` and get the experimental results for 20 rounds
        - move three result folders to Result-and-Vis
    - Result-and-Vis/
        - We already have the results for this paper (Cora, Citeseer, Pubmed). Users could choose to reproduce it by running train.py
        - run ```overfitting-vis.ipynb``` to generate the figures

## Citation
If you feel our paper and the code is useful. Please add the following citation. Contact <chaoqiy2@illinois.edu> for any question.
```bash
@article{yang2020revisiting,
  title={Revisiting over-smoothing in deep GCNs},
  author={Yang, Chaoqi and Wang, Ruijie and Yao, Shuochao and Liu, Shengzhong and Abdelzaher, Tarek},
  journal={arXiv preprint arXiv:2003.13663},
  year={2020}
}
```



