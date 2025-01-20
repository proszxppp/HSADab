# HSADab

![Fig1](https://github.com/user-attachments/assets/e1b07dfe-c603-4eba-a6d4-b09b9ec4ff02)

HSADab is the most comprehensive database for binding thermodynamics and all-atom structures of human serum albumin. HSA is the most prevalent protein found in plasma, lymph, saliva, cerebrospinal fluid, and interstitial fluid, and covers 60% of the total protein mass in human plasma. A unique feature of HSA is the large number of binding pockets, which makes it possible for external agents of diverse features to be encapsulated. 

![图片4](https://github.com/user-attachments/assets/49d24ad2-987a-4110-8491-5ffd52f59b2f)

The database is secured through an extensive literature review of more than 40,000 published contents relevant to HSA, covering 1987 to 2024.06. 

![图片1](https://github.com/user-attachments/assets/d255d396-5d80-472d-881a-85985fd30ecd)

The three pillars of HSADab are affinity, structure and docking banks, which makes it possible to grab some understanding about the structure-affinity relationship. The affinity bank contains binding thermodynamics of several thousand ligands towards HSA, with multiple temperature labels available. The structure bank contains all experimentally deposited HSA-related biomacromolecules, including not only the apo form and the ligand- or antibody-bound forms. The docking bank is constructed with the best local docking protocol PLANTS, the deep-learning tool DiffDock, and the current SOTA integrative deep-learning structural predictors AlphaFold3, Boltz-1 and Chai-1. Due to the huge size of the docking bank, here we only provide the PLANTS and DiffDock predictions, while those of AlphaFold3, Boltz-1 and Chai-1 are freely available in our HSADab webserver http://www.hsadab.cn/ or the dropbox links given below.

![redock_RMSD_all_lig](https://github.com/user-attachments/assets/e728d865-3b81-423e-ba51-fb03de300515)

A worth noting phenomenon is the underperformance of AutoDock families in redocking experiments, which adds caution to its common usage in published HSA-ligand studies. 

![ML_architectures](https://github.com/user-attachments/assets/7d8ef4aa-e02c-46bd-be30-154b25fa0cb9)

HSADab additionally supports affinity predictions with machine-learning predictors (the HSA_dG_pred dir). Among graph neural networks, large language models, and fingerprints-based regression models, the most robust and cost-effective one seems to be the regression models built on the fingerprints, which are thus provided in the HSA_dG_pred dir for batch predictions. 

![RMSE_MAE](https://github.com/user-attachments/assets/2effc1a7-19cd-4621-b95b-e769a0287996)

A fully interactive webserver for HSADab is available at http://www.hsadab.cn/. 

![图片5](https://github.com/user-attachments/assets/4faa9ce5-f541-4c27-9e68-7527e07cc7a2)

Dropbox links for all data banks of HSADab (v2025)

1. Affinity bank

https://www.dropbox.com/scl/fi/lk959iq7qfo79er51nrzt/affinity-bank-sep-mol.zip?rlkey=2y1xsx10cvl9q05ll931lh3u7&st=mme4uxyq&dl=0

2. Structure bank

https://www.dropbox.com/scl/fi/sylm57pbshso7cc78tgrg/structure-bank.zip?rlkey=6ez6lsc623sxp4qgongyf9zki&st=m26mhsc5&dl=0

3. Docking bank

3.1. AlphaFold3

https://www.dropbox.com/scl/fi/ifxb90j5hbxfumfuv0llj/AlphaFold3.7z?rlkey=o9159znlznnaosp03o668jm6r&st=m840cds6&dl=0

3.2. Boltz-1

https://www.dropbox.com/scl/fi/arkls3g9gq4gllwgub6bg/Boltz-1.7z?rlkey=tvbor8alpfe51m1y6nqkstq8q&st=1xamrdjy&dl=0

3.3. Chai-1

https://www.dropbox.com/scl/fi/1708bxxlce4bks9q65bvu/Chai-1.7z?rlkey=urwzsqdkigyn2vm63avacgpj1&st=ift42pu6&dl=0

3.4. DiffDock

https://www.dropbox.com/scl/fi/tb20wgrhw6ssizj940vtq/DiffDock.zip?rlkey=qra5gf67ge9lzm3sembfla137&st=ddujcnst&dl=0

3.5. PLANTS-chemplp

https://www.dropbox.com/scl/fi/7ilev2w4cn376wcmhlwdt/PLANTS_chemplp.tar.gz?rlkey=0lygv6lzpwk10etfu48ifro3n&st=opkgxl6n&dl=0

3.6. PLANTS-plp

https://www.dropbox.com/scl/fi/dij8f2c9egg4jk8qtepal/PLANTS_plp.tar.gz?rlkey=g1h3g972tygvdrlgkivdmblv7&st=wg1sazf9&dl=0
