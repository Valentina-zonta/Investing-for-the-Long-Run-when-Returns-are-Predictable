# Quantitative Asset Allocation Analysis

This repository contains the code, analysis, and final report for a comprehensive project on quantitative asset allocation. The project employs statistical methods, primarily regression analysis, to model and understand the behavior of financial assets.

The core of the project is a "literate programming" approach, where the main analysis is conducted in a Jupyter Notebook (`Main.ipynb`). This notebook imports and utilizes custom-built Python modules (`Regression_function.py`, `fun_not_fun.py`) that contain the project's core logic and utility tools. The final methodology, findings, and conclusions are formally presented in the `asset_allocation_report.pdf`.

## Project Components

This repository is organized into four key components, each serving a distinct purpose:

### 1. `asset_allocation_report.pdf`

This document is the final, human-readable output of the project. It provides a detailed, formal discussion of:

* **Introduction & Objectives:** The problem statement and the goals of the analysis.
* **Theoretical Background:** The financial and statistical theories underpinning the models used.
* **Methodology:** A step-by-step description of the data collection, data processing, and modeling techniques employed.
* **Results & Analysis:** The synthesized findings from the code, including key visualizations and statistical outputs.
* **Conclusion:** A summary of the project's findings and their implications.

For a complete understanding of the project's "why" and its conclusions, please refer to this report.

### 2. `Main.ipynb`

This Jupyter Notebook is the central "workbook" for the entire analysis. It serves as an executable narrative, guiding the user through the project from start to finish. Its key responsibilities include:

* **Data Ingestion:** Loading the raw financial data.
* **Data Preprocessing:** Cleaning, transforming, and preparing the data for modeling, often using functions imported from `fun_not_fun.py`.
* **Model Implementation:** Calling the pre-defined models from `Regression_function.py` to run the analysis on the prepared data.
* **Visualization & Interpretation:** Generating all the plots, tables, and statistical summaries necessary to understand the model outputs.
* **Exploratory Data Analysis (EDA):** Initial investigation of the data to inform modeling choices.

This notebook is the main entry point for replicating the analysis and exploring the results interactively.

### 3. `Regression_function.py`

This is a custom-built Python module that contains all the core logic for the project's regression models. By centralizing the complex logic in this file, the main notebook (`Main.ipynb`) remains clean, readable, and focused on the analysis rather than on code implementation.

This module likely contains:
* Functions to build, train, and test various regression models.
* Classes or functions for specific statistical tests or financial calculations.
* Logic for handling model parameters and returning structured results (e.g., coefficients, p-values, R-squared).

### 4. `fun_not_fun.py`

This is a utility module containing various helper functions that are used throughout the analysis. Its name suggests it handles the necessary but less "glamorous" tasks required for the project.

This script's purpose is to promote code reusability and keep the main notebook tidy. It likely includes functions for:
* Data cleaning (e.g., handling missing values, `NaNs`).
* Data transformation (e.g., calculating log returns, standardizing data).
* Specific financial calculations (e.g., calculating volatility, Sharpe ratios, or other metrics).
* Plot formatting helpers to ensure consistent visualizations.

## How to Use This Project

To replicate this analysis, please follow the steps below.

### 1. Prerequisites

You must have a Python environment (v3.7+) with the standard scientific computing stack. The primary dependencies are:

* `jupyter`
* `numpy`
* `pandas`
* `matplotlib`
* `scikit-learn`
* `statsmodels` (or other libraries used in your regression script)

### 2. Setup

First, clone this repository to your local machine:

```bash
git clone [URL_TO_THIS_REPOSITORY]
cd [REPOSITORY_FOLDER]
