# 🕵️ Analyzing Crime Data

![High Tech Detective Work](https://i.ibb.co/CPMP78d/DALL-E-2023-12-18-12-44-43-A-detailed-and-sophisticated-illustration-of-a-digital-crime-data-analysi.png)

## 📜 About
This Machine Learning Engineering lab traverses from meticulous data cleaning 🧹 to deep exploratory analysis 🔍, yielding nuanced insights into Chicago's crime data. Culminating with a polished XGBoost model 💡, enhanced by step-wise Hyperopt tuning and Recursive Feature Elimination, the project boasts an noteable 89% precision rate 🎯.

## 🗃️ Dataset Attribution
- Crime Data provided by Chicago Police Department. [View Dataset](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2)
- Census Data supplied by U.S. Census Bureau. [View Dataset](https://data.cityofchicago.org/Health-Human-Services/Census-Data-Selected-socioeconomic-indicators-in-C/kn9c-c2s2)

## ⚠️ Disclaimer
This project is an academic exercise. Content is for educational use and should not be considered as professional advice.

## 📋 Table of Contents
- [EDA Laboration](#eda-laboration)
  - Preface: Approach to Structuring Responses
  - Dataset Introduction
  - Warming up
  - Cleaning up the mess
  - The Bird's Eye
  - Chicago Police Department performance assessment
  - Troubles at home
  - Bad Boys Bad Boys whatcha gonna do
  - Night Stalker
  - Grand Theft Auto
  - Just send me like location
  - The 5 factor
  - Spotlight on you, Maestro!
- [Machine Learning](#machine-learning)
  - Project Overview: Predictive Modeling for Non-Arrest Incidents
  - Metric Focus: Precision in Predicting Non-Arrests
    - Objective
    - Importance of Non-Arrest Predictions
    - Precision-Driven Strategy
  - Exploratory Data Analysis (EDA)
    - Data Summary
    - Continuous Feature Distributions
    - Categorical & Discrete Feature Distributions
  - Scaler/Encoding Selection and Preprocessing
  - Feature Extraction and Selection
  - Data Split: Train, Validation, Test
  - XGBoost CV
  - Feature Importance Analysis
  - Model Evaluation
  - RFE Feature Selection
  - Model Update
  - Hyperparameter Tuning
    - Preprocessing: Resampling and New Data Split
    - HyperOpt Step-wise Tuning
    - Exhaustive Tuning Insights
  - Model Fitting and Tuning Recap
    - XGBoost with Early Stopping
    - XGBoost with GridSearchCV
  - Project Summary and Final Model Evaluation
    - Key Outcomes
    - Challenges and Learnings
    - Forward-Looking Improvements
    - Conclusion
    - A moment of reflection

## 💻 Installation and Usage
To run the Jupyter Notebook (`*.ipynb`) included in this project, you will need an environment capable of executing Python code and rendering Jupyter Notebooks. Options include:
- **JupyterLab / Jupyter Notebook**: Install locally via [Anaconda](https://www.anaconda.com/products/individual), which includes most data science libraries.
- **Google Colab**: Run the notebook in the cloud on [Google Colab](https://colab.research.google.com/) with no installation required.

Packages required are numerous and installation is covered step-by-step as you progress through the notebook.

## 📄 License
This project is released under the MIT License - see the [LICENSE.md](LICENSE) file for details.

## 🙏 Acknowledgments
- Special thanks to Ali Leylani for providing this valuable opportunity and sharing his vast reservoir of knowledge. His unwavering support, willingness to answer questions, and ability to clarify complex topics have been instrumental to the success of this project.
- Gratitude for the Chicago Police Department and U.S. Census Bureau for providing the datasets.
