# Exploratory Data Analysis (EDA) Portfolio

This repository contains comprehensive Exploratory Data Analysis projects covering three diverse datasets: Flight Price Prediction, Google Play Store Apps, and Wine Quality Assessment. Each project demonstrates different aspects of data analysis, from data cleaning and preprocessing to statistical analysis and visualization.

## 📊 Projects Overview

### 1. Flight Price Prediction Analysis (`flightprice_02.ipynb`)

**Dataset**: Flight booking data with 10,683 records and 11 features  
**Source**: `flightPrice/flight_price.xlsx`  
**Objective**: Analyze flight pricing patterns and prepare data for price prediction models

#### Key Features Analyzed:
- **Airlines**: Multiple airline carriers
- **Routes**: Source and destination cities
- **Timing**: Departure and arrival times, journey dates
- **Duration**: Flight duration and number of stops
- **Pricing**: Target variable for prediction

#### Data Processing Steps:
1. **Date Feature Engineering**: 
   - Extracted day, month, year from journey dates
   - Converted date components to integer format
2. **Time Feature Engineering**:
   - Split departure and arrival times into hour and minute components
   - Handled time format inconsistencies
3. **Categorical Encoding**:
   - Applied One-Hot Encoding to airline, source, and destination variables
   - Mapped stop categories to numerical values (non-stop=0, 1 stop=1, etc.)
4. **Duration Processing**:
   - Extracted hours and minutes from duration strings
   - Converted to separate numerical features
5. **Data Cleaning**:
   - Handled missing values in Route and Total_Stops columns
   - Removed redundant columns after feature extraction

#### Final Dataset Structure:
- **Original**: 11 columns, 10,683 rows
- **Processed**: 16+ columns with engineered features, ready for machine learning

---

### 2. Google Play Store Apps Analysis (`GooglePlayStore_03.ipynb`)

**Dataset**: Google Play Store app data with 10,841 records and 13 features  
**Source**: Raw CSV from GitHub repository  
**Objective**: Comprehensive analysis of mobile app ecosystem and market trends

#### Key Features Analyzed:
- **App Information**: Name, category, type (free/paid)
- **Performance Metrics**: Ratings, reviews, installs
- **Technical Details**: Size, content rating, Android version requirements
- **Market Data**: Price, last update information

#### Data Cleaning & Preprocessing:
1. **Data Quality Issues**:
   - Removed duplicate app entries (kept first occurrence)
   - Fixed data type inconsistencies in Reviews column
   - Handled missing values across multiple features
2. **Feature Engineering**:
   - Converted Reviews from string to integer
   - Standardized Size format (M/k to numerical values)
   - Cleaned Price and Installs columns (removed $, +, commas)
   - Extracted year, month, day from Last Updated dates
3. **Data Export**: Created cleaned dataset (`cleanDataCSV/google_cleaned.csv`)

#### Key Insights:
- **App Distribution**: Family apps (18.2%) and Games (10.6%) dominate the market
- **Pricing**: 92.6% of apps are free
- **Content Rating**: 80.4% apps rated "Everyone"
- **Market Analysis**: Category-wise installation patterns and rating distributions

#### Visualizations Created:
- Univariate analysis of numerical features (KDE plots)
- Categorical feature distributions (count plots)
- Top 10 app categories (pie chart and bar plot)
- Market share analysis by category

---

### 3. Wine Quality Assessment (`Winequality_01.ipynb`)

**Dataset**: Portuguese Vinho Verde wine quality data with 1,599 records  
**Source**: `wine+quality/winequality-red.csv`  
**Objective**: Analyze physicochemical properties and their relationship with wine quality

#### Dataset Information:
- **Wine Type**: Red wine samples from Portuguese "Vinho Verde"
- **Attributes**: 11 physicochemical properties + quality rating
- **Quality Scale**: 0-10 (based on expert sensory evaluation)

#### Features Analyzed:
1. **Acidity Metrics**: Fixed acidity, volatile acidity, citric acid
2. **Sugar Content**: Residual sugar
3. **Chemical Properties**: Chlorides, sulfur dioxide (free/total)
4. **Physical Properties**: Density, pH, sulphates, alcohol content
5. **Target Variable**: Quality score (0-10)

#### Analysis Techniques:
1. **Data Quality**:
   - Removed duplicate records (1599 → 1359 records)
   - Verified no missing values
2. **Statistical Analysis**:
   - Descriptive statistics for all features
   - Correlation analysis with heatmap visualization
   - Distribution analysis using histograms
3. **Quality Assessment**:
   - Imbalanced dataset analysis (quality distribution)
   - Feature correlation with quality scores
   - Box plots for quality vs. alcohol content

#### Key Findings:
- **Quality Distribution**: Dataset is imbalanced with most wines rated 5-6
- **Feature Correlations**: Strong relationships between certain chemical properties
- **Quality Factors**: Alcohol content and other physicochemical properties influence quality ratings

---

## 🛠️ Technical Stack

### Libraries Used:
- **Data Manipulation**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn` (OneHotEncoder)
- **Data Import**: `openpyxl` (for Excel files)

### Key Techniques Demonstrated:
1. **Data Cleaning**: Missing value handling, duplicate removal, data type conversion
2. **Feature Engineering**: Date/time extraction, categorical encoding, string manipulation
3. **Statistical Analysis**: Correlation analysis, descriptive statistics, distribution analysis
4. **Visualization**: Heatmaps, histograms, box plots, bar charts, pie charts
5. **Data Export**: Clean dataset creation for further analysis

---

## 📁 Repository Structure

```
EDA/
├── README.md                          # This file
├── flightprice_02.ipynb              # Flight price analysis
├── GooglePlayStore_03.ipynb          # Google Play Store analysis
├── Winequality_01.ipynb              # Wine quality analysis
├── flightPrice/
│   └── flight_price.xlsx             # Original flight data
├── cleanDataCSV/
│   └── google_cleaned.csv            # Cleaned Google Play Store data
└── wine+quality/
    ├── winequality-red.csv           # Red wine dataset
    ├── winequality-white.csv         # White wine dataset
    └── winequality.names             # Dataset documentation
```

---

## 🎯 Key Learning Outcomes

### Data Science Skills Demonstrated:
1. **Data Exploration**: Understanding dataset structure and quality
2. **Data Cleaning**: Handling real-world messy data
3. **Feature Engineering**: Creating meaningful features from raw data
4. **Statistical Analysis**: Correlation analysis and distribution understanding
5. **Visualization**: Creating informative plots for data storytelling
6. **Data Preparation**: Preparing datasets for machine learning models

### Domain Knowledge Gained:
- **Aviation Industry**: Flight booking patterns and pricing factors
- **Mobile App Market**: App store ecosystem and user behavior
- **Wine Industry**: Physicochemical properties affecting wine quality

---

## 🚀 Getting Started

### Prerequisites:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

### Running the Analyses:
1. Clone or download this repository
2. Ensure all data files are in their respective folders
3. Open Jupyter notebooks in order:
   - Start with `Winequality_01.ipynb` (simplest dataset)
   - Move to `GooglePlayStore_03.ipynb` (medium complexity)
   - End with `flightprice_02.ipynb` (most complex feature engineering)

### Data Sources:
- **Flight Data**: Local Excel file
- **Google Play Store**: Public dataset from GitHub
- **Wine Quality**: UCI Machine Learning Repository

---

## 📈 Future Enhancements

### Potential Extensions:
1. **Machine Learning Models**: Build prediction models using the processed data
2. **Advanced Visualizations**: Interactive dashboards using Plotly or Dash
3. **Statistical Testing**: Hypothesis testing and confidence intervals
4. **Time Series Analysis**: Trend analysis for temporal data
5. **Clustering**: Unsupervised learning techniques for market segmentation

### Additional Analyses:
- Cross-dataset comparisons
- Advanced feature selection techniques
- Outlier detection and treatment
- Model performance evaluation

---

## 📞 Contact

This EDA portfolio demonstrates proficiency in data analysis, statistical thinking, and data storytelling. Each project showcases different aspects of the data science workflow, from initial data exploration to final insights and recommendations.

**Note**: This portfolio is designed for educational and demonstration purposes, showcasing comprehensive exploratory data analysis skills across diverse domains.
