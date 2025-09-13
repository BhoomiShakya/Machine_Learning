# Feature Engineering Notebooks

A comprehensive collection of Jupyter notebooks demonstrating essential feature engineering techniques for machine learning.

## 📚 Notebooks Overview

| Notebook | Description | Key Topics |
|----------|-------------|------------|
| **Handling_Missing_value.ipynb** | Missing value detection and imputation techniques | MCAR, MAR, MNAR, Mean/Median/Mode imputation, Deletion methods |
| **Hnadling_oulier.ipynb** | Outlier detection and handling strategies | IQR method, Z-score, Box plots, 5-number summary |
| **Label_Enchoding.ipynb** | Label encoding for categorical variables | Ordinal data encoding, sklearn LabelEncoder |
| **One_Hot_Encoding.ipynb** | One-hot encoding implementation | Nominal categorical variables, binary vectors |
| **Ordinal_encoding.ipynb** | Ordinal encoding with custom categories | Hierarchical data, ordered categories |
| **Target_guided.ipynb** | Target-guided encoding techniques | Mean encoding, high-cardinality variables |
| **Handling_Imbalance_Dataset.ipynb** | Imbalanced dataset handling methods | Random oversampling, class imbalance solutions |
| **SMOTE.ipynb** | SMOTE implementation and visualization | Synthetic minority oversampling, class balancing |

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Feature-Engineering
   ```

2. **Install required packages**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
   ```

3. **Run the notebooks**
   - Open Jupyter Notebook or JupyterLab
   - Navigate to any notebook and run all cells
   - Each notebook is self-contained with examples and explanations

## 📋 Prerequisites

- Python 3.7+
- Jupyter Notebook/Lab
- Required packages (see requirements.txt)

## 🛠️ Key Libraries Used

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning preprocessing
- **matplotlib/seaborn**: Data visualization
- **imbalanced-learn**: Imbalanced dataset handling

## 📖 Learning Path

1. **Start with**: `Handling_Missing_value.ipynb` - Understanding data quality issues
2. **Then**: `Hnadling_oulier.ipynb` - Detecting and handling outliers
3. **Continue with**: Categorical encoding notebooks (Label, One-Hot, Ordinal, Target-guided)
4. **Finish with**: Imbalanced dataset handling (`Handling_Imbalance_Dataset.ipynb` and `SMOTE.ipynb`)

## 🎯 What You'll Learn

- **Data Quality**: How to identify and handle missing values and outliers
- **Categorical Encoding**: Different methods for converting categorical data to numerical format
- **Class Imbalance**: Techniques to handle imbalanced datasets
- **SMOTE**: Advanced synthetic sampling techniques
- **Best Practices**: Industry-standard approaches to feature engineering

## 📊 Key Techniques Covered

### Missing Value Handling
- Deletion methods (listwise, pairwise)
- Imputation methods (mean, median, mode)
- Advanced techniques (KNN, iterative imputation)

### Outlier Detection
- Statistical methods (IQR, Z-score)
- Visualization techniques (box plots)
- Handling strategies (removal, capping, transformation)

### Categorical Encoding
- **Label Encoding**: For ordinal variables
- **One-Hot Encoding**: For nominal variables
- **Ordinal Encoding**: For hierarchical data
- **Target-Guided Encoding**: For high-cardinality variables

### Imbalanced Data Handling
- **Oversampling**: Random oversampling, SMOTE
- **Undersampling**: Random undersampling
- **Evaluation Metrics**: Precision, Recall, F1-Score, ROC-AUC

## 🔧 Usage Examples

Each notebook contains:
- **Theory**: Explanation of concepts
- **Code Examples**: Practical implementations
- **Visualizations**: Charts and plots for better understanding
- **Best Practices**: Industry recommendations
- **Summary**: Key takeaways

## 📈 Performance Tips

- Always split data before applying transformations
- Use cross-validation for robust evaluation
- Document your feature engineering pipeline
- Consider computational costs of different techniques

## 🤝 Contributing

Feel free to:
- Report issues or bugs
- Suggest new techniques to add
- Improve existing examples
- Add more comprehensive explanations

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 📞 Contact

For questions or suggestions, please open an issue in this repository.

---

**Happy Learning! 🎓**

*Master feature engineering to build better machine learning models.*

