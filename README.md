
# Solana Price Data Analysis Repository

This repository contains a comprehensive analysis of Solana price data collected from **01-01-2021** to **29-09-2024** in OHLCV format. The project demonstrates the process of fetching raw data, adding technical indicators, performing data cleaning, feature engineering, and preprocessing to ultimately generate an enhanced dataset with **44 columns**.

---

## Repository Structure

- **Datasets:**
  - `solana_price_dataset.csv`  
    *The original OHLCV dataset containing basic open, high, low, close, and volume data.*
  - `solana_price_dataset_with_indicators.csv`  
    *The dataset after technical indicators have been added using the `ta` library.*

- **Notebooks:**
  - `add_indicators.ipynb`  
    *Jupyter Notebook for fetching the OHLCV data and adding technical indicators.*
  - `data_analysis.ipynb`  
    *Jupyter Notebook for performing data cleaning, feature engineering, and preprocessing.*

- **Figures:**
  - `./images/solana_price_chart.png`  
    *A sample visualization of the Solana price data with added indicators.*

---

## Notebooks Overview

### add_indicators.ipynb

- **Purpose:**  
  This notebook demonstrates how to:
  - Fetch Solana price data in OHLCV format.
  - Add various technical indicators using the `ta` library.
  - Save the resulting dataset as `solana_price_dataset_with_indicators.csv`.

- **Key Library:**  
  [ta](https://github.com/bukosabino/ta)

### data_analysis.ipynb

- **Purpose:**  
  This notebook is used to:
  - Clean the dataset.
  - Perform feature engineering.
  - Execute preprocessing steps in an orderly manner.
  - Produce the final, enhanced dataset with 44 columns.

---

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/your_username/solana-price-data-analysis.git
cd solana-price-data-analysis
```


