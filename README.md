# OpenPoly: A Benchmark Suite for Polymer Property Prediction

**OpenPoly** is a comprehensive, literature-derived benchmark platform designed to advance machine learning research in polymer science. It supports multi-property prediction across diverse chemical structures by providing a high-quality dataset of 3,985 polymer entries, each annotated with up to 26 experimentally measured properties. The benchmark facilitates the fair comparison of various learning paradigms under realistic, data-limited conditions commonly encountered in polymer informatics.

---

## Repository Structure

This repository offers all essential components for training, validating, and benchmarking deep learning models on polymer property prediction tasks:

- **`data/`**  
  Contains curated datasets aligned with polymer backbones encoded via PSMILES. The dataset covers 26 distinct properties, totaling 3,985 structure–property pairs.

- **`model/`**  
  Includes PyTorch implementations of benchmarked models. External model sources used in this repository:
  - [UniMol2](https://github.com/deepmodeling/Uni-Mol/tree/main/unimol2)
  - [polyBERT](https://github.com/Ramprasad-Group/polyBERT)
  - [TabPFN](https://github.com/PriorLabs/TabPFN)

- **`results/`**  
  Stores quantitative performance metrics (Mean Squared Error, Mean Absolute Error, and Coefficient of Determination) for each model–property pair.

- **`train_and_test_code/`**  
  Provides reproducible training and evaluation scripts, including data preprocessing, hyperparameter tuning, and model evaluation protocols.

---

## Model Availability

Due to the size of trained model files, all pre-trained weights (MLP, GCN, GAT, SphereNet, UniMol2, polyBERT, TabPFN, and XGBoost across 26 properties) are hosted on Zenodo and can be accessed via the following DOI:

**[10.5281/zenodo.15551637](https://doi.org/10.5281/zenodo.15551637)**

---

## Citation

If you use OpenPoly in your research, please cite:

> Wang, J. et al. OpenPoly: A Literature-Derived Polymer Database Empowering Benchmarking and Multi-Property Prediction. *In preparation*.

---

## License

This project is released under the MIT License. See the [`LICENSE`](./LICENSE) file for full terms.