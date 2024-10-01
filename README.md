# HSADab

![overview_HSA2025](https://github.com/user-attachments/assets/6d782646-f6ae-42e8-9cda-e10882bb92e7)

HSADab is the most comprehensive database for binding thermodynamics and all-atom structures of human serum albumin. HSA is the most prevalent protein found in plasma, lymph, saliva, cerebrospinal fluid, and interstitial fluid, and covers 60% of the total protein mass in human plasma. A unique feature of HSA is the large number of binding pockets, which makes it possible for external agents of diverse features to be encapsulated. 

![图片4](https://github.com/user-attachments/assets/49d24ad2-987a-4110-8491-5ffd52f59b2f)

The database is secured through an extensive literature review of more than 40,000 published contents relevant to HSA, covering 1987 to 2024.06. 

![图片3](https://github.com/user-attachments/assets/1f55b867-58fe-473c-822d-ade8259d7b50)

The three pillars of HSADab are affinity, structure and docking banks, which makes it possible to grab some understandings about the structure-affinity relationship.  
The affinity bank contains binding thermodynamics of several thousand ligands towards HSA, with multiple temperature labels available. The structure bank contains all experimentally deposited HSA-related biomacromolecules, including not only the apo form and the ligand- or antibody-bound forms. The docking bank is constructed with the best local docking protocol PLANTS and the deep-learning tool DiffDock. 

A worth noting phenomenon is the underperformance of AutoDock families in redocking experiments, but such treatment is quite common in published HSA-ligand studies. 

![redock_RMSD_all_lig](https://github.com/user-attachments/assets/e728d865-3b81-423e-ba51-fb03de300515)

HSADab additionally supports affinity predictions with machine-learning predictors (the HSA_dG_pred dir). Among graph neural networks, large language models, and fingerprints-based regression models, the most robust and cost-effective one seems to be the regression models built on the fingerprints, which are thus provided in the HSA_dG_pred dir for batch predictions. 

![RMSE_MAE](https://github.com/user-attachments/assets/2effc1a7-19cd-4621-b95b-e769a0287996)

A fully interactive webserver for HSADab is available on http://www.hsadab.cn/. 

![图片5](https://github.com/user-attachments/assets/4faa9ce5-f541-4c27-9e68-7527e07cc7a2)
