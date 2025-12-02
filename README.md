# Hosted app: **[ProjeCT 360 on Streamlit](https://ai4all-crime-prediction-aiupyyt2azfwf4rhp7tuem.streamlit.app/)**  



---


# 🔍 CT-Crimes — AI4ALL Ignite Crime Prediction Project

**CT-Crimes**: AI-Driven Crime Prediction & Analysis
CT-Crimes is a machine learning initiative investigating how artificial intelligence can predict and analyze reported crime trends across Connecticut. Developed during the AI4ALL Ignite Fellowship by Kenneth Maeda, Manushri Pendekanti, and Min Thaw Zin, this project leverages real-world, publicly available data to build an ethical, explainable AI system aligned with the mission of advancing AI for social good.

---

## 🎯 Purpose & Outcomes
Our goal was to uncover meaningful spatial and temporal crime patterns by engineering a robust, end-to-end machine learning pipeline for public safety data. We applied rigorous data preprocessing, advanced model optimization, and interpretability techniques to create a tool that empowers communities and decision-makers with actionable insights for proactive safety measures and data-driven interventions.

Applied data preprocessing, model optimization, and interpretability techniques to build an ethical, explainable AI system aligned with **AI4ALL Ignite’s mission** of advancing AI for social good.

---

## 🧩 Problem Statement
Crime is a complex social issue affecting safety, resource allocation, and policy. However, predictive tools for localized crime analysis are often inaccessible, opaque, or limited in scope. CT-Crimes bridges this gap by building an accessible, interpretable model that forecasts regional crime trends, fostering accountability and informed decision-making.

---

## 📊 Key Results
- Dual-Model Architecture: Designed a system comprising two distinct LightGBM models:
  - Volume Forecaster (Regression): Predicts the count of daily incidents.
  - Risk Classifier (Multi-class): Predicts the type of crime most likely to occur given specific spatial/temporal contexts.

- Recursive Time-Series Forecasting: Implemented a "blind" recursive forecasting loop for the volume model to prevent data leakage, ensuring that future predictions rely only on past predictions rather than actual future data.

- Data Integration & Engineering: Unified over 497,000 records from 2021–2024, merging disparate datasets to calculate normalized metrics like crime rates per 1,000 residents and officer-to-population ratios.

- Resource Analytics Module: Engineered specific logic to track and visualize police force demographics, providing a granular breakdown of officer counts by gender across 95+ municipalities.

- Interactive Streamlit Dashboard: Deployed a fully interactive interface featuring:
  - Volume Forecasting: 30-day trend lines with holiday indicators.
  - Risk Analysis: Dynamic comparison of predicted crime probabilities vs. historical crime distributions (Pie vs. Bar charts).
  - Officer Trends: Toggleable views for statewide vs. city-specific force analysis with stacked demographic visualization.
    
---

## ⚙️ Methodologies
To achieve our goals, the team executed a comprehensive workflow:

- Pipeline Design: Created an end-to-end pipeline for data ingestion, cleaning, and transformation, specifically handling the propagation of 2024 officer statistics across historical records for consistent resource analysis.
  
- Feature Engineering: Developed "lag" features (past crime counts), rolling averages, and calendar-based attributes (holidays, day of week) to capture temporal dependencies.
  
- Bias Mitigation: Applied SMOTE (Synthetic Minority Oversampling Technique) to effectively mitigate severe class imbalances in crime type data.

- Model Evaluation: Conducted rigorous "Global Blind Backtests" to validate the volume forecaster's performance on unseen data, achieving a global volume accuracy of ~76%.

---

## Video Demo



https://github.com/user-attachments/assets/8bc9d641-bd9a-4302-8397-f8d309ef6046




---

## 🧠 Data Sources
- [FBI Crime Data Explorer](https://cde.ucr.cjis.gov/LATEST/webapp/#/pages/downloads): Crime Incident-Based Data by State (Connecticut 2021 - 2024)
---

## 🛠️ Technologies Used
**Languages:** Python  
**Libraries & Frameworks:** pandas, NumPy, scikit-learn, TensorFlow/Keras  
**Models:** Deep Neural Network (DNN), Random Forest, LightGBM, Logistic Regression

**Visualization:** Streamlit, Matplotlib, Plotly, Seaborn  
**Techniques:** SMOTE (imbalanced-learn)


---

## 🎓 Learning Outcomes
Through this project, we gained valuable technical and research experience across the full machine learning lifecycle — from data acquisition to deployment and bias evaluation. Key takeaways include:

* **End-to-End ML Development:** Learned how to design, train, and evaluate machine learning models (LightGBM, Random Forest, DNNs) on large-scale structured datasets.  

* **Data Ethics & Bias Analysis:** Understood the ethical implications of AI in law enforcement, identifying and mitigating sources of data bias through comparative evaluation and fairness metrics.  

* **Feature Engineering & Model Optimization:** Practiced transforming raw data into meaningful features, tuning hyperparameters, and balancing model complexity with interpretability.  

* **Data Visualization & Communication:** Strengthened skills in presenting technical findings clearly to both technical and non-technical audiences using visual analytics and dashboards.  

* **Collaborative Research:** Worked within the **AI4ALL Ignite** mentorship framework, collaborating with peers and mentors to conduct research aligned with real-world AI for social good principles.  

* **Practical AI Deployment:** Built an interactive **Streamlit app** for real-time prediction and visualization — bridging the gap between data science prototypes and accessible public tools.  

> These experiences reinforced our belief that **responsible AI is not just about accuracy, but about accountability, transparency, and impact**.

## 👥 Authors
This project was developed collaboratively under the **AI4ALL Ignite Fellowship**:

- **Kenneth Maeda** — [GitHub](https://github.com/NoVaxiion) 
- **Manushri Pendekanti** — [GitHub](https://github.com/manushrip06)
- **Min Thaw Zin** - [GitHub](https://github.com/Min-13)

---

## ⚖️ Educational Use
Developed for educational and research purposes as part of the **AI4ALL Ignite Fellowship**, promoting **ethical, interpretable, and community-centered AI** in public data analysis.
