# HSADab

![overview_HSA2025](https://github.com/user-attachments/assets/6d782646-f6ae-42e8-9cda-e10882bb92e7)

HSADab is the most comprehensive database for binding thermodynamics and all-atom structures of human serum albumin. 

The database is secured through an extensive literature review of more than 40,000 published contents relevant to HSA, covering 1987 to 2024.06. 
The three pillars of HSADab are affinity, structure and docking banks, which makes it possible to grab some understandings about the structure-affinity relationship.  
HSADab additionally supports affinity predictions with machine-learning predictors (the HSA_dG_pred dir). Among graph neural networks, large language models, and fingerprints-based regression models, the most robust and cost-effective one seems to be the random forest model built on the fingerprints, which is thus provided in the HSA_dG_pred dir for batch predictions. 

A fully interactive webserver for HSADab is available on http://www.hsadab.cn/. 

