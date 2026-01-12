# 🚀 Advanced RUL Analysis Toolkit for NASA C-MAPSS

## Research Project CYS6001-20: Aircraft Engine Remaining Useful Life Prediction

This comprehensive toolkit provides **publication-quality analysis** for evaluating deep learning models on the NASA C-MAPSS turbofan engine dataset. Designed to impress your professor with rigorous statistical analysis, beautiful visualizations, and cyber-security focused robustness testing.

---

## 🎯 What Makes This Toolkit Special

### 1. **Beyond Basic Metrics**
- Standard metrics (RMSE, MAE, R², PHM Score)
- **Confidence intervals** via bootstrapping
- **Statistical significance testing** (Wilcoxon, Friedman)
- **Effect size calculation** (Cohen's d)
- Error distribution analysis (skewness, kurtosis, Q-Q plots)

### 2. **Cyber-Security Focus** 🔒
- **Robustness to sensor noise** analysis
- Performance degradation quantification
- Adversarial resilience scoring
- Critical for aviation safety applications

### 3. **Publication-Quality Outputs**
- High-resolution (300 DPI) figures
- LaTeX-ready comparison tables
- Professional formatting
- Ready to insert into your paper

### 4. **Lifecycle Analysis**
- Performance across engine lifecycle stages
- Early life vs. late life comparison
- Critical phase (<20 cycles) evaluation

---

## 📁 Project Structure

```
├── rul_analysis_toolkit.py      # Core analysis framework
├── run_evaluation.py            # Main evaluation pipeline
├── save_predictions.py          # Helper for saving predictions
├── statistical_analysis.py      # Advanced statistical tests
├── quick_start_template.py      # Auto-generated starter code
│
├── data/                        # Place C-MAPSS dataset here
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   ├── RUL_FD001.txt
│   └── ... (FD002, FD003, FD004)
│
├── predictions/                 # Save your model predictions here
│   ├── SVR_FD001_predictions.npy
│   ├── LSTM_FD001_predictions.npy
│   ├── TCN_FD001_predictions.npy
│   └── Transformer_FD001_predictions.npy
│
└── results/                     # Auto-generated outputs
    ├── *.png                    # Visualizations
    ├── *.csv                    # Result tables
    ├── *.tex                    # LaTeX tables
    └── statistical_significance.csv
```

---

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy torch
```

### Step 2: Save Your Model Predictions

```python
from save_predictions import save_model_predictions

# After training your model and getting test predictions:
save_model_predictions(
    predictions=your_predictions_array,
    model_name='LSTM',  # or 'SVR', 'TCN', 'Transformer'
    fd_number=1  # or 3 for FD003
)
```

### Step 3: Run Comprehensive Evaluation

```bash
python run_evaluation.py
```

That's it! All results, visualizations, and tables will be in `./results/`

---

## 📊 What You Get

### 1. Performance Comparison Table
```
Model       RMSE   MAE    R2     PHM_Score  Early_Life_RMSE  Late_Life_RMSE
─────────────────────────────────────────────────────────────────────────────
SVR         20.45  15.32  0.68   2847.21    18.92            22.67
LSTM        16.82  12.43  0.78   1923.45    15.21            18.94
TCN         15.12  11.05  0.82   1456.78    14.03            16.89
Transformer 15.34  11.23  0.81   1512.34    14.28            17.12
```
*Exported as CSV and LaTeX*

### 2. Robustness Analysis (Cyber-Security)
```
Model       Clean_RMSE  Noisy_RMSE  Degradation%  Robustness_Score
───────────────────────────────────────────────────────────────────
SVR         20.45       25.67       25.5%         74.5/100
LSTM        16.82       19.23       14.3%         85.7/100
TCN         15.12       16.45       8.8%          91.2/100  ⭐
Transformer 15.34       19.89       29.7%         70.3/100
```

### 3. Statistical Significance Testing
```
Model 1      Model 2      p-value    Significant  Cohen's d  Effect Size
───────────────────────────────────────────────────────────────────────
TCN          Transformer  0.0234     ✓            0.612      medium
TCN          LSTM         0.0012     ✓            0.843      large
Transformer  LSTM         0.1234     ✗            0.234      small
```

### 4. Visualizations

#### Prediction vs. Actual
![](https://i.imgur.com/example1.png)
*Scatter plot with regression line, perfect prediction line, and metrics*

#### Error Distribution Analysis
![](https://i.imgur.com/example2.png)
*4-panel visualization: histogram with KDE, Q-Q plot, residual plot, boxplot*

#### Robustness Comparison
![](https://i.imgur.com/example3.png)
*Bar charts showing performance degradation and robustness scores*

#### Lifecycle Performance
![](https://i.imgur.com/example4.png)
*RMSE across different engine lifecycle stages*

---

## 🔬 Advanced Features

### 1. Bootstrap Confidence Intervals
```python
from rul_analysis_toolkit import RULAnalyzer

analyzer = RULAnalyzer('LSTM')
ci = analyzer.compute_confidence_intervals(y_true, y_pred, confidence=0.95)

print(f"RMSE: {rmse:.2f} ± {(ci['RMSE_CI'][1] - ci['RMSE_CI'][0])/2:.2f}")
# Output: RMSE: 15.12 ± 0.43
```

### 2. Statistical Significance Testing
```python
from statistical_analysis import generate_significance_report

# Compare all models with statistical rigor
significance_df = generate_significance_report(y_true, predictions)
# Includes: Wilcoxon tests, Friedman test, Cohen's d, Bonferroni correction
```

### 3. Cross-Validation Stability
```python
from statistical_analysis import CrossValidationAnalyzer

cv_results = {
    'LSTM': [16.5, 17.2, 16.8, 17.0, 16.9],  # 5-fold CV scores
    'TCN': [15.0, 15.3, 15.1, 15.2, 15.1],
    # ... other models
}

analyzer = CrossValidationAnalyzer()
stability_df = analyzer.analyze_cv_stability(cv_results)
# Shows: Mean, Std, CV coefficient, Min, Max, Range
```

### 4. Lifecycle Performance Analysis
```python
lifecycle_metrics = analyzer.early_vs_late_performance(y_true, y_pred, threshold=50)
# Compares RMSE/MAE for early life (>50 cycles) vs late life (≤50 cycles)
```

---

## 📝 Example Usage Scenarios

### Scenario 1: Quick Evaluation (Already Have Predictions)

```python
from run_evaluation import run_comprehensive_evaluation

# Just run this!
results, robustness = run_comprehensive_evaluation(
    fd_number=1,
    data_path='./data',
    predictions_path='./predictions'
)
```

### Scenario 2: Integrate with Your Training Loop

```python
import torch
from save_predictions import save_model_predictions

# Train your model
model = YourModel()
train_model(model, train_loader)

# Evaluate on test set
model.eval()
predictions = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        y_pred = model(X_batch)
        predictions.extend(y_pred.cpu().numpy())

# Save and analyze
save_model_predictions(np.array(predictions), 'YourModel', fd_number=1)

# Run full evaluation
from run_evaluation import run_comprehensive_evaluation
results, robustness = run_comprehensive_evaluation(fd_number=1)
```

### Scenario 3: Custom Analysis Pipeline

```python
from rul_analysis_toolkit import RULAnalyzer, VisualizationSuite

# Create analyzer
analyzer = RULAnalyzer('CustomModel')
viz = VisualizationSuite()

# Get comprehensive metrics
metrics = analyzer.compute_standard_metrics(y_true, y_pred)
ci = analyzer.compute_confidence_intervals(y_true, y_pred)
error_dist = analyzer.analyze_error_distribution(y_true, y_pred)

# Generate specific visualizations
viz.plot_prediction_vs_actual(y_true, y_pred, 'CustomModel', 'output.png')
viz.plot_error_distribution(y_true, y_pred, 'CustomModel', 'errors.png')
```

---

## 🎓 For Your Research Paper

### Tables to Include

1. **Table 1**: Performance Comparison (use `FD001_results_table.tex`)
2. **Table 2**: Robustness Analysis (use `FD001_robustness.csv`)
3. **Table 3**: Statistical Significance (use `statistical_significance.csv`)

### Figures to Include

1. **Figure 1**: Prediction vs. Actual (all models)
2. **Figure 2**: Error Distribution Analysis (best model)
3. **Figure 3**: Robustness Comparison Bar Chart
4. **Figure 4**: Lifecycle Performance Comparison

### Key Statements to Make

✅ "Confidence intervals computed via bootstrap resampling (n=1000)"
✅ "Statistical significance assessed using Wilcoxon signed-rank test"
✅ "Effect sizes calculated using Cohen's d"
✅ "Robustness quantified under controlled sensor noise (σ=0.01)"
✅ "All models evaluated on identical preprocessing pipeline"

---

## 🔍 Interpretation Guide

### RMSE Values
- < 12: Excellent
- 12-15: Very Good
- 15-18: Good
- 18-22: Acceptable
- > 22: Needs improvement

### Robustness Score
- > 90: Highly robust (deployment-ready)
- 75-90: Robust (acceptable with monitoring)
- 60-75: Moderate (requires safeguards)
- < 60: Vulnerable (not recommended)

### Cohen's d Effect Size
- < 0.2: Negligible difference
- 0.2-0.5: Small difference
- 0.5-0.8: Medium difference
- > 0.8: Large difference

### p-values
- < 0.001: Highly significant (***)
- < 0.01: Very significant (**)
- < 0.05: Significant (*)
- ≥ 0.05: Not significant (ns)

---

## 🎯 Key Features That Will Impress Your Professor

### 1. Statistical Rigor
- ✅ Bootstrap confidence intervals
- ✅ Non-parametric significance tests
- ✅ Effect size quantification
- ✅ Multiple comparison correction (Bonferroni)
- ✅ Power analysis capability

### 2. Cyber-Security Angle
- ✅ Adversarial robustness testing
- ✅ Sensor integrity analysis
- ✅ Availability attack simulation
- ✅ Deployment readiness scoring

### 3. Professional Presentation
- ✅ Publication-quality figures (300 DPI)
- ✅ LaTeX-ready tables
- ✅ Consistent color schemes
- ✅ Comprehensive error analysis

### 4. Reproducibility
- ✅ Fixed random seeds
- ✅ Version-controlled preprocessing
- ✅ Documented methodology
- ✅ Shareable code

---

## 🐛 Troubleshooting

### Issue: "File not found" errors
**Solution**: Make sure your directory structure matches the expected layout
```bash
mkdir -p data predictions results
```

### Issue: "Module not found" errors
**Solution**: Install all dependencies
```bash
pip install -r requirements.txt
```

### Issue: Predictions have wrong shape
**Solution**: Ensure predictions are 1D NumPy arrays
```python
# If 2D:
predictions = predictions.flatten()

# If list:
predictions = np.array(predictions)
```

### Issue: Getting NaN or Inf values
**Solution**: Check for:
- Division by zero in RUL calculation
- Negative predictions (clip to 0)
- Missing values in data

```python
predictions = np.nan_to_num(predictions, nan=0.0, posinf=1e6, neginf=0.0)
predictions = np.maximum(predictions, 0)  # Ensure non-negative
```

---

## 📚 References

This toolkit implements methods from:

1. **PHM Score**: Saxena & Goebel (2008) - NASA Prognostics Data Repository
2. **Bootstrap CI**: Efron & Tibshirani (1993) - An Introduction to the Bootstrap
3. **Effect Size**: Cohen (1988) - Statistical Power Analysis
4. **Wilcoxon Test**: Wilcoxon (1945) - Individual Comparisons by Ranking Methods
5. **Friedman Test**: Friedman (1937) - The Use of Ranks to Avoid Assumptions

---

## 🤝 Contributing

Found a bug or have a suggestion? This toolkit is designed for your research project. Feel free to modify and extend it for your specific needs!

---

## 📄 License

This toolkit is provided for academic research purposes. Please cite appropriately if used in publications.

---

## 💡 Tips for Maximum Impact

1. **Run on both FD001 and FD003** to show performance on simple and complex data
2. **Include statistical significance** - don't just report numbers, prove they're different!
3. **Emphasize robustness** - this is your cyber-security angle
4. **Show failure cases** - analyze which samples have highest errors and why
5. **Add cross-validation** - shows your results are reproducible
6. **Compare to baselines** - your deep models should beat SVR significantly

---

## 🎉 Good Luck!

This toolkit gives you everything needed for a comprehensive, statistically rigorous analysis that will impress your professor. Focus on:

1. **Clear methodology** - use this consistent pipeline
2. **Rigorous statistics** - confidence intervals and significance tests
3. **Practical implications** - robustness for deployment
4. **Professional presentation** - high-quality figures and tables

You've got this! 🚀

---

**Last Updated**: January 12, 2026  
**Version**: 1.0.0  
**Author**: Research Project CYS6001-20
