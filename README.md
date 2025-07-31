# AI-Driven Early Warning System for Sovereign Debt Default

This project provides a machine learning-based API that predicts the risk of sovereign debt default in developing countries using macroeconomic indicators.

---

## 🌍 Project Overview

Sovereign debt default is a significant economic risk, especially in developing nations. This project aims to provide an **early warning system** using machine learning to predict whether a country is at risk of defaulting on its debt.

---

## 📊 Features Used

The model takes in the following input features:

- `gdp_growth`: Annual GDP growth rate (%)
- `inflation_rate`: Annual inflation rate (%)
- `external_debt`: Total external debt as a percentage of GDP
- `foreign_reserves`: Total foreign reserves in USD billions
- `political_stability`: Index score (-2.5 to +2.5)

---

## 🤖 Model

A **Logistic Regression** model was trained on a synthetically generated dataset that simulates realistic economic scenarios in developing nations. The trained model is saved as `default_model.pkl`.

---

## 🚀 FastAPI Endpoint

### Base URL (local):
