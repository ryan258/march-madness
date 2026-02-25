# **Project: Automated NCAAM Bracketology & Predictive Modeling**

## **1\. Project Overview**

This project aims to build an end-to-end automated system that scrapes NCAA Men’s College Basketball (NCAAM) data via ESPN’s undocumented API endpoints to predict March Madness tournament outcomes. By combining high-velocity game data with predictive indices like ESPN’s Basketball Power Index (BPI), the model generates win probabilities for any 64-team bracket configuration.

## **2\. Technical Stack & Dependencies**

### **Core Automation & Scraping**

* **Python 3.9+**: Primary programming language.  
* **sdv-py (SportsDataverse)**: A Python wrapper for ESPN’s API, handling complex routing for play-by-play and box score data.  
* **Requests & BeautifulSoup4**: For direct targeting of JSON endpoints and supplementary HTML scraping.

### **Data Processing & ML**

* **Pandas & NumPy**: For data cleaning, feature engineering, and handling massive JSON payloads.  
* **Scikit-learn**: To implement Logistic Regression and Random Forest classifiers for matchup simulation.  
* **XGBoost (Optional)**: For high-performance gradient boosting to refine upset prediction accuracy.

## **3\. Data Architecture: ESPN API Endpoints**

The system targets three primary undocumented endpoints to feed the model:

| Data Type | Endpoint Path (Base: site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/) | Purpose |
| :---- | :---- | :---- |
| **Scoreboard** | /scoreboard | Retrieves daily game schedules, results, and real-time scores. |
| **Team Info** | /teams/{team\_id} | Accesses rosters, historical season stats, and specific team records. |
| **Rankings** | /rankings | Pulls current BPI and AP Top 25 rankings to establish baseline strength. |

## **4\. Feature Engineering Strategy**

The model's "Edge" is derived from blending raw stats with advanced metrics. We prioritize the following features:

* **Efficiency Ratio**: Adjusted Offensive Efficiency / Adjusted Defensive Efficiency.  
* **Momentum Factor**: Weighted performance over the last 10 games of the regular season.  
* **Strength of Schedule (SoS)**: Normalizing point differentials based on opponent BPI.  
* **Upset Propensity**: Identifying mid-major teams with high Effective Field Goal Percentage (eFG%) and low turnover rates.

## **5\. The Prediction Pipeline**

1. **Ingestion**: Automated script runs via sdv-py to pull 5-10 years of historical tournament data.  
2. **Training**: A Logistic Regression model is trained on historical matchups where the target variable is Team\_1\_Win (Boolean).  
3. **Simulation**: On Selection Sunday, the script pulls the new bracket, runs every possible matchup through the model, and outputs a CSV of win probabilities.  
4. **Optimization**: A recursive function maps probabilities to the bracket structure to identify the path with the highest expected value (EV).

## **6\. Project Milestones**

* **Phase 1**: Script development for automated API polling and JSON parsing.  
* **Phase 2**: Database construction (SQL or Parquet) containing historical regular-season performance.  
* **Phase 3**: ML Model training and cross-validation against 2018-2024 tournament results.  
* **Phase 4**: Frontend visualization (Streamlit) to display the "Optimized Bracket."

## **7\. Ethical & Technical Disclaimers**

* **API Stability**: ESPN’s internal APIs are undocumented and subject to change without notice. The system includes robust error handling and backoff logic.  
* **Use Case**: This model is for research and entertainment purposes; "guaranteed" brackets are statistically impossible due to the high variance of single-elimination formats.