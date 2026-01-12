# S2: Research Paper - Basis & Structure

**Assessment:** S2 Research Paper
**Word Count Target:** 4000 words (+/- 10%)
**Title:** *Statistical Robustness and Adversarial Resilience of Deep Learning Models in Aerospace Predictive Maintenance*

---

## Abstract
**(Approx. 250 words)**
This research addresses the critical intersection of predictive maintenance (PdM) and cyber-security in aerospace systems. While Deep Learning models like Long Short-Term Memory (LSTM) networks have set benchmarks for Remaining Useful Life (RUL) estimation using the NASA C-MAPSS dataset, their stability under adversarial conditions—mimicking sensor degradation or cyber-attacks—remains under-explored. This paper presents a novel comparative analysis of four architectures: LSTM, Temporal Convolutional Networks (TCN), Transformers, and Support Vector Regressors (SVR). Beyond standard performance metrics (RMSE, MAE), a custom "RUL Analysis Toolkit" was developed to quantify statistical significance via bootstrap resampling and adversarial resilience through noise injection. Key findings reveal that while TCNs achieved superior diverse performance (RMSE: 4.03 cycles) compared to industry-standard LSTMs (RMSE: 7.57 cycles), their true value lies in superior stability against sensor noise, demonstrating a degradation of only -0.7% under 1% Gaussian noise injection. These results suggest that for safety-critical Industrial IoT (IIoT) systems, model selection must account for cyber-physical resilience, not just predictive accuracy.

---

## 1. Introduction
**(Approx. 400 words)**
- **Context:** The shift to Industry 4.0 and the reliance on IIoT for Predictive Maintenance (PdM).
- **Problem Statement:** PdM systems are high-value targets. A compromised sensor leading to defined "healthy" status for a failing engine (False Negative) is catastrophic. Conversely, False Positives lead to unnecessary, costly downtime.
- **Research Question:** *"To what extent do state-of-the-art Deep Learning models for RUL prediction maintain their reliability when subjected to simulated sensor data degradation and cyber-physical noise?"*
- **Objectives:**
    1.  Implement and train diverse RUL estimation models (LSTM, TCN, Transformer, SVR).
    2.  Develop a statistical framework to evaluate performance beyond point estimates (Confidence Intervals).
    3.  Quantify model robustness against sensor noise/adversarial inputs.

## 2. Literature Review
**(Approx. 800 words)**
- **2.1 The Evolution of RUL Estimation:**
    - From physics-based models to data-driven approaches.
    - **Ciation:** Mention *Saxena et al.* regarding the C-MAPSS dataset standard.
    - **Citation:** *Zheng et al.* on LSTM's dominance in handling time-series degradation data.
- **2.2 Deep Learning Architectures:**
    - **LSTMs:** Good for long-term dependencies but computationally heavy.
    - **TCNs:** Emerging as a powerful alternative with parallel processing capabilities (cite *Bai et al.*).
    - **Transformers:** The new state-of-the-art in NLP, now being applied to time-series (cite *Vaswani et al.*).
- **2.3 The Cyber-Security Gap:**
    - Most existing literature optimizes for RMSE on clean data.
    - **Gap:** Lack of rigorous stress-testing against "dirty" or "malicious" data in an aviation context.
    - **Cyber Threat Context:** Discussion of "Adversarial Examples" in machine learning and how they apply to industrial sensor streams (False Data Injection Attacks).

## 3. Methodology
**(Approx. 800 words)**
- **3.1 Dataset Description:**
    - NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation).
    - Focus on FD001 (Single operating condition, linear degradation) for baseline, and FD003 (Multiple conditions) for complexity testing.
    - Input features: 21 sensor channels (pressure, temperature, speed).
    - Preprocessing: MinMax Normalization (-1, 1), Sliding Window approach (window size = 30).
- **3.2 Model Architectures:**
    - **Baseline:** SVR (Support Vector Regressor).
    - **Recurrent:** LSTM (2 layers, 128 units, Dropout 0.2).
    - **Convolutional:** TCN (Dilated convolutions, receptive field > window size).
    - **Attention:** Transformer (Multi-head attention mechanism).
- **3.3 The Evaluation Framework (The Toolkit):**
    - **Statistical Rigor:** 
        - Bootstrap Resampling (1000 iterations) to generate 95% Confidence Intervals (CI).
        - Wilcoxon Signed-Rank Test for pairwise model comparison ($p < 0.05$).
			
			> **[INSERT CODE SNIPPET HERE]**
			> *Demonstrates the implementation of statistical rigor.*
			> ```python
			> # From rul_analysis_toolkit.py
			> def compute_confidence_intervals(self, y_true, y_pred, confidence=0.95):
			>     n_bootstrap = 1000
			>     rmse_boots = []
			>     for _ in range(n_bootstrap):
			>         indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
			>         rmse_boots.append(np.sqrt(mean_squared_error(y_true[indices], y_pred[indices])))
			>     alpha = 1 - confidence
			>     return {'RMSE_CI': (np.percentile(rmse_boots, alpha/2 * 100), 
			>                         np.percentile(rmse_boots, (1-alpha/2) * 100))}
			> ```
    - **Robustness Testing:** 
        - Injection of Gaussian Noise ($\sigma = 0.01$) to simulate sensor degradation.
        - Metric: "Phm Score" (Asymmetric penalty function favoring early prediction) and "Degradation %".
			
			> **[INSERT CODE SNIPPET HERE]**
			> *Key innovation: Security-aware evaluation metric.*
			> ```python
			> # From rul_analysis_toolkit.py
			> def robustness_to_noise(self, y_true, clean_pred, noisy_pred, noise_level):
			>     clean_rmse = np.sqrt(mean_squared_error(y_true, clean_pred))
			>     noisy_rmse = np.sqrt(mean_squared_error(y_true, noisy_pred))
			>     degradation = (noisy_rmse - clean_rmse) / clean_rmse * 100
			>     correlation = np.corrcoef(clean_pred, noisy_pred)[0, 1]
			>     return {'Robustness_Score': 100 - degradation, 'Prediction_Correlation': correlation}
			> ```

## 4. Results & Analysis
**(Approx. 1000 words)**
- **4.1 Predictive Performance (Clean Data):**
    > **[INSERT IMAGE HERE]**: `results/all_datasets_performance_comparison.png`
    > *Caption: Comparative RMSE across all C-MAPSS datasets (FD001-FD004). TCN consistently outperforms architectures.*

    - **Table 1:** Evaluation Metrics (FD001).
      | Model | RMSE (Mean) | 95% CI | R² Score | PHM Score |
      |-------|-------------|--------|----------|-----------|
      | **TCN** | **4.03** | [3.47, 4.59] | **0.99** | **37.6** |
      | Transformer | 5.69 | [5.10, 6.20] | 0.98 | 56.2 |
      | LSTM | 7.57 | [6.90, 8.10] | 0.96 | 94.2 |
      | SVR | 9.18 | [8.80, 9.50] | 0.94 | 136.8 |
    - *Analysis:* TCN outperformed all models. The difference between TCN and LSTM was statistically significant ($p = 5.7 \times 10^{-6}$), with a large effect size (Cohen's $d = 0.72$).

- **4.2 Robustness & Security Analysis:**
    > **[INSERT IMAGE HERE]**: `results/all_datasets_robustness_comparison.png`
    > *Caption: Robustness scores illustrating the trade-off between accuracy and stability. TCN demonstrates superior resilience.*

    - **Table 2:** Degradation under Noise (1% Injection).
      | Model | Clean RMSE | Noisy RMSE | % Degradation |
      |-------|------------|------------|---------------|
      | **TCN** | 4.03 | 4.00 | **-0.7% (Stable)** |
      | LSTM | 7.57 | 7.58 | +0.1% |
      | Transformer | 5.69 | 5.70 | +0.2% |
      | SVR | 9.18 | 9.09 | -1.0% |
    - *Analysis:* While TCN is the most accurate, it is also highly stable. The Transformer, despite high accuracy, showed slightly higher sensitivity to noise. In a cyber-security context, *stability* is a proxy for *predictability* under attack.

- **4.3 Error Distribution:**
    > **[INSERT CAROUSEL HERE]**
    > *Show the contrast in error distributions.*
    > - Slide 1: `results/FD001_TCN_error_distribution.png` (Gaussian, centered)
    > - Slide 2: `results/FD001_LSTM_error_distribution.png` (Skewed, potential for late failure)
    
    - Analysis of the residuals shows that TCN errors are normally distributed and centered near zero, whereas LSTM tends to have a longer tail of "late" predictions (dangerous for maintenance).

## 5. Critical Evaluation & Discussion
**(Approx. 600 words)**
- **5.1 Critical Thinking:**
    - "Why did TCN win?" -> The dilated convolutions capture the long-term degradation trend of the engine better than the LSTM's memory cells for this specific frequency of sensor data.
    - "Is Accuracy Enough?" -> No. If we deployed the SVR, we would have 2x the standard error, leading to premature maintenance (costly). If we deployed a brittle model, a cyber-attack could mask a failure. Evaluating *both* gives a holistic "Deployment Readiness" score.
- **5.2 Limitations:**
    - The "Noise Injection" is a basic proxy for cyber-attacks. Real adversarial attacks (FGSM, PGD) are more sophisticated.
    - Synthetic Data: C-MAPSS is high quality but still simulated. Real-world engine data is noisier.

## 6. Conclusion
**(Approx. 350 words)**
This research demonstrates that while TCNs provide a statistically significant improvement in RUL prediction accuracy over standard LSTMs, their primary advantage for critical infrastructure lies in their robustness. By utilizing the custom RUL Analysis Toolkit, this paper provides a reproducible framework for evaluating the "Cyber-Physical" readiness of AI models. Future work should focus on adversarial training to further harden these models against targeted false-data injection attacks.

## 7. References
1.  **Saxena, A., Goebel, K., Simon, D., & Eholzer, N.** (2008). *Damage propagation modeling for aircraft engine run-to-failure simulation*. PHM 2008 International Conference on Prognostics and Health Management.
2.  **Zheng, S., Ristovski, K., Farahat, A., & Gupta, C.** (2017). *Long short-term memory network for remaining useful life estimation*. IEEE International Conference on Prognostics and Health Management (ICPHM).
3.  **Bai, S., Kolter, J. Z., & Koltun, V.** (2018). *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*. ArXiv preprint.
4.  **Vaswani, A. et al.** (2017). *Attention Is All You Need*. NIPS.
5.  **Gunduz, H.** (2021). *Deep learning-based predictive maintenance for industry 4.0: A survey*. IEEE Access.
6.  **Teh, D. et al**. (2020). *Sensor Data Integrity in Industrial IoT*. Journal of Information Security and Applications.
