# S1: Lightning Talk - Presentation Basis

**Assessment:** S1 Lightning Talk
**Time Limit:** 5 Minutes (Strict)
**Goal:** Showcase the essence of the research (Context, Methodology, Findings, Importance).

---

## Slide 1: Title & Introduction
**Visual:**
- Title: **"Securing Prediction: Robust RUL Estimation for Aircraft Engines against Cyber-Physical Threats"**
- Subtitle: Advanced Statistical Analysis & Adversarial Resilience
- Your Student ID
- Bath Spa University Logo

**Speaker Notes (0:00 - 0:45):**
"Good [morning/afternoon]. My research addresses a critical vulnerability in modern aerospace maintenance: the reliability of AI-driven predictive systems under cyber-physical stress.

While industry standard models like LSTMs can accurately predict when an aircraft engine needs maintenance—known as Remaining Useful Life or RUL—my research asks a simpler, more dangerous question: **Can we trust these predictions if the sensor data is compromised?**

Today I will present a novel analysis toolkit I developed that doesn't just measure accuracy, but quantifies *robustness* against cyber-threats."

---

## Slide 2: Research Aim & Context
**Visual:**
- A split screen image:
    - Left: A clean sensor signal (Standard Operation).
    - Right: A noisy/injected sensor signal (Cyber Threat/Degradation).
- Key Question: *"How does sensor integrity impact safety-critical maintenance decisions?"*
- Bullet points:
    - **Gap:** Most studies focus only on RMSE (Error).
    - **My Focus:** Error Stability + Adversarial Resilience.

**Speaker Notes (0:45 - 2:00):**
"Predictive maintenance systems, or PdM, rely on the NASA C-MAPSS dataset standard. Current literature is obsessed with minimizing the Root Mean Squared Error (RMSE). If a model predicts an engine has 50 cycles left, and it actually has 52, that's great accuracy.

However, in an Industrial IoT context, these sensors are attack surfaces. My research question was: **How robust are state-of-the-art models (like TCNs and LSTMs) when subjected to data degradation that mimics cyber-attacks, such as signal injection or denial of service?**

I hypothesized that the models with the highest raw accuracy might actually be the most brittle when facing these anomalies. To answer this, I moved beyond basic metrics to develop a statistical framework for *security-aware* evaluation."

---

## Slide 3: Methodology (The Toolkit)
**Visual:**
- A diagram of your `RUL Analysis Toolkit` pipeline:
    1.  **Input:** C-MAPSS Data (FD001-FD004).
    2.  **Models:** LSTM, TCN, Transformer, SVR.
    3.  **The "Stress Test":**
        - Bootstrap Resampling (1000 iter).
        - Noise Injection (Adversarial simulation).
    4.  **Output:** Statistical Significance (Wilcoxon/Friedman).

**Speaker Notes (2:00 - 3:15):**
"To test this, I built a comprehensive Python analysis toolkit.
I evaluated four distinct architectures: Support Vector Regressors, LSTMs, Temporal Convolutional Networks (TCNs), and Transformers.

But I didn't just run them once. I implemented **Bootstrap Confidence Intervals** with 1,000 iterations to prove that my results weren't just a fluke.
Crucially, I developed a 'Robustness Module' that injects Gaussian noise and bias into the test data—simulating sensor degradation or subtle cyber-manipulation—and measures how much the RUL prediction degrades.

This effectively creates a 'Cyber-Resilience Score' for each AI architecture."

---

## Slide 4: Findings & Results
**Visual:**
- **Bar Chart:** Comparison of TCN vs LSTM vs SVR.
    - Highlight TCN as "Best Performance" (RMSE ~4.03).
- **Table snippet:** Statistical Significance (p-values < 0.05).
- **Key Insight Graphic:** "TCN Degradation: -0.7%" vs "Transformer: +0.2%".

**Speaker Notes (3:15 - 4:15):**
"The results were revealing.
Statistically, the **Temporal Convolutional Network (TCN)** was the superior model, achieving a Root Mean Squared Error of just 4.03 cycles. Using the Wilcoxon signed-rank test, I confirmed this improvements was statistically significant with a p-value less than 0.001 against the industry-standard LSTM.

But the most interesting finding came from the resilience testing. While the Transformer model was highly accurate, it showed different stability characteristics under noise compared to the TCN. The TCN demonstrated remarkable robustness, maintaining its predictive capability even with 1% signal corruption, making it the ideal candidate for hostile or noisy operational environments."

---

## Slide 5: Importance & Conclusion
**Visual:**
- **Impact:**
    1.  **Safety:** Preventing catastrophic failure from false negatives.
    2.  **Security:** Detecting model drift under attack.
- **Final Takeaway:** *"Accuracy is nothing without Resilience."*

**Speaker Notes (4:15 - 5:00):**
"Why does this matter?
In aviation, a false prediction isn't just a number—it's safety-critical. If an attacker can manipulate sensor data to make an engine look healthy when it's failing, the consequences are disastrous.

My research proves that we cannot select models based on accuracy alone. We must evaluate their **statistical stability and cyber-resilience**. My toolkit provides a standardized way to do exactly that, ensuring the next generation of predictive maintenance systems are not just smart, but secure.

Thank you."
