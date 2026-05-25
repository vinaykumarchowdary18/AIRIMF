# AIRIMF – AI-Driven Risk Identification and Mitigation Framework

This repository contains the full experimental pipeline, datasets, and figures for the paper **"AIRIMF: An AI-Driven Risk Identification and Mitigation Framework for Software Project Management"**.

## Repository Structure

- `data/` : Raw and processed CSV files (VS Code and React bug reports, ablation results).
- `src/` : All Python scripts for scraping, feature engineering, model training, SHAP/ablation analysis, and figure generation.
- `figures/` : Publication-ready figures (feature importance, confusion matrix, SHAP summary).
- `paper/` : LaTeX source for the IEEE manuscript.

## How to Reproduce the Results

1. **Clone the repository.**
2. **Install dependencies:**
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn textblob shap lightgbm matplotlib seaborn requests
   # VS Code data already included. To re-scrape React:
export GITHUB_TOKEN='your_token_here'
python src/scrape_react.py
python src/airimf_v4_shap_ablation_lightgbm.py
python src/airimf_figure_generator.py
cd paper
pdflatex AIRIMF
pdflatex AIRIMF
@inproceedings{airimf2025,
  author    = {Mandadi Vinay Kumar},
  title     = {AIRIMF: An AI-Driven Risk Identification and Mitigation Framework for Software Project Management},
  booktitle = {Proceedings of ...},
  year      = {2026}
}

---

## 🔄 Summary of File Mapping

| Current File | Destination |
|--------------|-------------|
| `vscode_bugs_datasets.csv` | `data/` |
| `react_bugs_raw.csv` | `data/` |
| `react_bugs_dataset.csv` | `data/` |
| `AIRIMF_Engineered_Dataset.csv` | `data/` |
| `AIRIMF_Final_Balanced_Dataset.csv` | `data/` |
| `test_project_risks.csv` | `data/` |
| `ablation_study_results.csv` | `data/` |
| `scrape_react.py` | `src/` |
| `github_scraper_test.py` | `src/` |
| `airimf_data_engineer.py` | `src/` |
| `airimf_model_training.py` | `src/` |
| `airimf_master_pipeline.py` | `src/` |
| `airimf_v3_advanced_pipeline.py` | `src/` |
| `airimf_v3_react_pipeline.py` | `src/` |
| `airimf_v4_shap_ablation_lightgbm.py` | `src/` |
| `airimf_figure_generator.py` | `src/` |
| `smote_augmentation.py` | `src/` |
| `IEEE_Fig2_Feature_Importance.png` | `figures/` |
| `IEEE_Confusion_Matrix.png` | `figures/` |
| `IEEE_SHAP_Explanation.png` | `figures/` |
| `AIRIMF.tex` (final version) | `paper/` |

Now your research is fully reproducible, transparent, and ready for GitHub. Push it, and you’ll have a professional repository to share with reviewers and the community.
