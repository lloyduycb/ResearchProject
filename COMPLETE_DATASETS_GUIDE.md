# Complete Guide: All C-MAPSS Datasets (FD001-FD004)

## 📚 Dataset Overview

### FD001 - Simple Baseline ★☆☆
- **Operating Conditions**: Single (Sea level)
- **Fault Modes**: Single (HPC Degradation)
- **Training Units**: 100 engines
- **Test Units**: 100 engines
- **Use Case**: Baseline performance, algorithm development
- **Complexity**: LOW

### FD002 - Multiple Conditions ★★☆
- **Operating Conditions**: Six different flight conditions
- **Fault Modes**: Single (HPC Degradation)
- **Training Units**: 260 engines
- **Test Units**: 259 engines
- **Use Case**: Test adaptability to operating condition variations
- **Complexity**: MEDIUM

### FD003 - Multiple Faults ★★☆
- **Operating Conditions**: Single (Sea level)
- **Fault Modes**: Two (HPC + Fan Degradation)
- **Training Units**: 100 engines
- **Test Units**: 100 engines
- **Use Case**: Test ability to distinguish between fault types
- **Complexity**: MEDIUM

### FD004 - Maximum Complexity ★★★
- **Operating Conditions**: Six different flight conditions
- **Fault Modes**: Two (HPC + Fan Degradation)
- **Training Units**: 248 engines
- **Test Units**: 249 engines
- **Use Case**: Real-world scenario with maximum variability
- **Complexity**: HIGH

---

## 🎯 How to Use This Toolkit with All Datasets

### Option 1: Analyze One Dataset at a Time

```python
# For FD001 (simplest)
python run_evaluation.py  # default is FD001

# For FD003 (multiple faults)
# Edit run_evaluation.py: change fd_number=1 to fd_number=3
python run_evaluation.py
```

### Option 2: Analyze All Datasets at Once (RECOMMENDED)

```bash
python run_all_datasets.py
```

This will:
1. ✅ Analyze all four datasets (FD001-FD004)
2. ✅ Generate individual results for each
3. ✅ Create cross-dataset comparisons
4. ✅ Analyze complexity impact
5. ✅ Provide overall model rankings

---

## 📊 Expected Results Pattern

### Typical RMSE Progression (Lower is Better)

```
Dataset    SVR      LSTM     TCN      Transformer
─────────────────────────────────────────────────
FD001     18-22    14-17    12-15    13-16      ← Easiest
FD002     22-28    18-24    16-20    17-21
FD003     20-25    16-20    14-18    15-19
FD004     24-32    20-28    18-24    19-25      ← Hardest
```

**Key Pattern**: 
- RMSE increases with complexity (FD001 → FD004)
- Deep models (LSTM/TCN/Transformer) handle complexity better than SVR
- TCN typically shows best balance of accuracy and robustness

---

## 🔬 What Makes Each Dataset Important

### FD001: Validation of Core Algorithm
- **Purpose**: Prove your model works in ideal conditions
- **Paper Section**: "5.1 Performance on FD001 (Clean Data)"
- **What to Show**: Absolute best performance numbers
- **Key Statement**: "Under single operating condition, TCN achieved RMSE of X.XX..."

### FD002: Operating Condition Robustness
- **Purpose**: Show model adapts to different flight regimes
- **Paper Section**: "5.2 Performance on Complex Data"
- **What to Show**: Comparison with FD001, degradation analysis
- **Key Statement**: "When varying operating conditions, model maintained Y% of baseline performance..."

### FD003: Fault Discrimination Capability
- **Purpose**: Prove model distinguishes between fault types
- **Paper Section**: "5.2 Performance on Complex Data" or separate subsection
- **What to Show**: Similar to FD001 but with fault mode discussion
- **Key Statement**: "With multiple fault modes, model demonstrated ability to..."

### FD004: Real-World Validation
- **Purpose**: Most realistic scenario - proves deployment readiness
- **Paper Section**: "6. Discussion"
- **What to Show**: Comprehensive analysis, deployment recommendation
- **Key Statement**: "Under realistic operating conditions with multiple fault modes (FD004), the recommended TCN model achieved..."

---

## 📝 Recommended Paper Structure Using All Datasets

### 5. Results and Analysis

#### 5.1 Baseline Performance (FD001)
- Present main comparison table with all models
- Include prediction vs actual plot
- Include error distribution analysis
- Show statistical significance

**Figure 1**: Model comparison on FD001  
**Table 1**: Performance metrics (FD001)  
**Table 2**: Statistical significance tests  

#### 5.2 Performance on Complex Data (FD003 or FD004)
- Show how performance changes with complexity
- Emphasize which models degrade gracefully
- Include robustness analysis

**Figure 2**: Model comparison on FD003/FD004  
**Table 3**: Performance metrics (FD003/FD004)  

#### 5.3 Cross-Dataset Analysis
- Show performance progression FD001→FD004
- Analyze generalization capability
- Overall model rankings

**Figure 3**: Performance across all datasets  
**Figure 4**: Complexity impact analysis  
**Table 4**: Cross-dataset summary  

### 6. Discussion and Critical Thinking

#### 6.1 Robustness to Sensor Noise (ALL DATASETS)
- Show robustness scores across all datasets
- Analyze consistency of robustness
- Cyber-security implications

**Figure 5**: Robustness comparison across datasets  
**Table 5**: Robustness summary  

#### 6.2 Implications for Deployment
- Overall recommendation based on all datasets
- Discuss trade-offs (accuracy vs robustness vs complexity)
- Deployment considerations

---

## 🎯 Directory Structure for All Datasets

```
project/
├── data/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   ├── RUL_FD001.txt
│   ├── train_FD002.txt
│   ├── test_FD002.txt
│   ├── RUL_FD002.txt
│   ├── train_FD003.txt
│   ├── test_FD003.txt
│   ├── RUL_FD003.txt
│   ├── train_FD004.txt
│   ├── test_FD004.txt
│   └── RUL_FD004.txt
│
├── predictions/
│   ├── SVR_FD001_predictions.npy
│   ├── LSTM_FD001_predictions.npy
│   ├── TCN_FD001_predictions.npy
│   ├── Transformer_FD001_predictions.npy
│   ├── SVR_FD002_predictions.npy
│   ├── ... (repeat for FD002, FD003, FD004)
│
└── results/
    ├── FD001_results.csv
    ├── FD002_results.csv
    ├── FD003_results.csv
    ├── FD004_results.csv
    ├── all_datasets_rmse_summary.csv
    ├── all_datasets_robustness_summary.csv
    ├── model_rankings.csv
    └── *.png (all visualizations)
```

---

## 💡 Pro Tips for Each Dataset

### FD001 Tips
✅ Use this for initial algorithm validation  
✅ Get your best absolute numbers here  
✅ Perfect for detailed error analysis  
✅ Use for statistical significance tests  

### FD002 Tips
✅ Emphasize operating condition adaptability  
✅ Compare preprocessing strategies  
✅ Discuss feature importance across conditions  
✅ Show your model isn't overfitting to single condition  

### FD003 Tips
✅ Emphasize fault discrimination capability  
✅ Analyze which fault mode is harder to predict  
✅ Discuss feature patterns for different faults  
✅ Show model distinguishes between degradation types  

### FD004 Tips
✅ This is your "real-world" validation  
✅ Emphasize generalization capability  
✅ Use for deployment recommendations  
✅ Show robust performance under maximum complexity  

---

## 📊 Key Metrics to Report for Each Dataset

### Must Report for ALL Datasets
1. RMSE (primary metric)
2. MAE (interpretability)
3. R² (goodness of fit)
4. PHM Score (asymmetric penalty)

### Report Once (Usually FD001)
1. Statistical significance tests
2. Detailed error distribution
3. Confidence intervals
4. Effect sizes

### Report in Cross-Dataset Analysis
1. Average RMSE across all datasets
2. Performance degradation (FD001 → FD004)
3. Robustness consistency
4. Overall rankings

---

## 🎓 Sample Paper Statements Using All Datasets

### Introduction
```
"We evaluate our models across all four C-MAPSS sub-datasets (FD001-FD004), 
representing increasing levels of operational complexity, from single 
operating condition with single fault mode (FD001) to multiple operating 
conditions with multiple fault modes (FD004)."
```

### Methodology
```
"To ensure comprehensive evaluation, we tested each model on all four 
C-MAPSS datasets, varying in complexity from the baseline FD001 to the 
most challenging FD004 scenario."
```

### Results
```
"On the baseline FD001 dataset, TCN achieved an RMSE of X.XX (95% CI: 
Y.YY-Z.ZZ), significantly outperforming the SVR baseline (p < 0.001, 
d = 0.85, large effect). When evaluated on the most complex FD004 
dataset, TCN maintained robust performance (RMSE: A.AA), demonstrating 
only a B.B% degradation compared to FD001."
```

### Discussion
```
"Cross-dataset analysis revealed that TCN exhibited the most consistent 
performance across all operating conditions and fault modes, with an 
average RMSE of X.XX across FD001-FD004. This generalization capability, 
combined with superior robustness scores (avg: 91.2/100), makes TCN the 
recommended architecture for deployment in real-world aviation maintenance 
systems."
```

---

## ✅ Checklist for Complete Analysis

### Data Preparation
- [ ] Downloaded all C-MAPSS datasets (FD001-FD004)
- [ ] Placed in correct directory structure
- [ ] Verified file formats and contents

### Model Training
- [ ] Trained models on FD001 training set
- [ ] Trained models on FD002 training set (if evaluating FD002)
- [ ] Trained models on FD003 training set
- [ ] Trained models on FD004 training set (if evaluating FD004)

### Prediction Generation
- [ ] Generated predictions for all models on FD001 test
- [ ] Generated predictions for all models on FD003 test
- [ ] (Optional) Generated for FD002 and FD004
- [ ] Saved in correct format (.npy files)

### Analysis Execution
- [ ] Ran individual dataset evaluations
- [ ] Ran cross-dataset comparison
- [ ] Generated all visualizations
- [ ] Created summary tables

### Paper Integration
- [ ] Selected key figures (4-6 figures max)
- [ ] Selected key tables (3-5 tables max)
- [ ] Wrote results section with statistical support
- [ ] Emphasized cross-dataset generalization
- [ ] Discussed robustness implications

---

## 🚀 Quick Start Commands

### Minimum Analysis (FD001 + FD003)
```bash
# Analyze simplest and multi-fault datasets
python run_evaluation.py  # FD001 by default

# Edit run_evaluation.py: fd_number=3
python run_evaluation.py  # FD003
```

### Comprehensive Analysis (All Datasets)
```bash
# One command to rule them all!
python run_all_datasets.py
```

### Just Generate Demo Results
```bash
# See what outputs look like
python demo.py
```

---

## 📈 Expected Timeline

### Minimum Viable Paper (FD001 only)
- Data prep: 1 hour
- Training: 2-4 hours
- Analysis: 15 minutes
- **Total: ~1 day**

### Good Paper (FD001 + FD003)
- Data prep: 1 hour
- Training: 4-8 hours
- Analysis: 30 minutes
- **Total: ~1-2 days**

### Excellent Paper (All FD001-FD004)
- Data prep: 2 hours
- Training: 8-16 hours
- Analysis: 1 hour
- **Total: ~2-3 days**

---

## 🎯 Bottom Line

**Minimum for passing**: FD001 only  
**Recommended for good grade**: FD001 + FD003  
**Recommended for excellent grade**: All datasets with cross-analysis  

The toolkit supports all approaches - choose based on your:
- Time available
- Computational resources
- Desired paper depth
- Target grade

**Good luck!** 🚀
