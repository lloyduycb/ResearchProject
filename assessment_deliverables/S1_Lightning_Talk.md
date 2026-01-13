# S1: Lightning Talk - Presentation Script
**Assessment:** S1 Lightning Talk
**Time Limit:** 5 Minutes (Strict)
**Structure:** 10 Slides (Strict User Template)

---

## Slide 1: Title
**Visual:**
- **Title:** Statistical Robustness and Adversarial Resilience of Deep Learning Models in Aerospace Predictive Maintenance
- **Student Name/ID**
- **Module:** Research Project (CYS6001-20)

**Speaker Notes (0:00 - 0:20):**
"Good [morning/afternoon]. My name is [Name], and my research project is titled 'Statistical Robustness and Adversarial Resilience of Deep Learning Models in Aerospace Predictive Maintenance'. 
This study investigates how we can secure AI-driven engineering systems against the growing threat of cyber-physical attacks."

---

## Slide 2: Inspiration Behind Your Research Topic
**Visual:**
- A collage or split image: 
    - *Personal/Academic:* A laptop running code (Data Science interest).
    - *Societal:* A news headline about a cyber-attack on infrastructure or the SolarWinds hack.
- **Key Motivation:** "The intersection of AI innovation and Cyber-Security vulnerability."

**Speaker Notes (0:20 - 0:50):**
"My inspiration for this topic came from a glaring gap I observed between two fields.
Academically, I was fascinated by the power of Deep Learning to predict engine failures. But societally, we are seeing an explosion in attacks on critical infrastructure—from the Colonial Pipeline to GPS spoofing.
I realized that while we are building smarter engines, we might be building *less secure* ones. I wanted to explore this urgent intersection: ensuring that our AI pilots aren't just smart, but unhackable."

---

## Slide 3: Introduction
**Visual:**
- **Theme:** "From Schedule to Prediction"
- **Context:**
    - Traditional: Replace at 10,000 hours (Safe but Wasteful).
    - Predictive: Replace when efficient (Optimal but Risky).
- **Core Problem:** Predictive systems trust sensors implicitly.

**Speaker Notes (0:50 - 1:20):**
"To set the stage: Aviation is shifting from 'Scheduled Maintenance'—replacing parts blindly after a set number of hours—to 'Predictive Maintenance', where AI analyzes sensor data to calculate Remaining Useful Life (RUL).
This shift is worth billions. However, it introduces a dangerous assumption: that the sensor data is always true. 
My research addresses the problem of **Sensor Integrity**. If an attacker injects noise or biases a sensor, standard AI models can fail catastrophically. This matters because a single false prediction can ground a fleet or endanger a flight."

---

## Slide 4: Research Objectives
**Visual:**
- **Objective 1:** **Benchmark** diverse Deep Learning architectures (LSTM, TCN, Transformer).
- **Objective 2:** **Quantify** Adversarial Resilience (Stability under noise).
- **Objective 3:** **Validate** Physics-based feature importance.

**Speaker Notes (1:20 - 1:50):**
"To address this, I set three clear objectives:
First, to implement and benchmark a diverse range of architectures—LSTMs, Temporal Convolutional Networks, and Transformers—on the NASA C-MAPSS dataset.
Second, to go beyond accuracy and quantifying their 'Adversarial Resilience'—measuring exactly how much their predictions degrade when the data is corrupted.
And third, to validate that these models are learning real physical degradation patterns, not just statistical noise."

---

## Slide 5: Research Questions
**Visual:**
- **Question:** *"To what extent do TCNs outperform recurrent architectures (LSTMs) in adversarial IIoT environments?"*
- **Hypothesis:** *"TCNs will demonstrate superior constituent stability due to their non-recursive, filter-based architecture."*

**Speaker Notes (1:50 - 2:15):**
"This led to my primary research question:
*To what extent do Temporal Convolutional Networks (TCNs) outperform industry-standard LSTMs when subjected to adversarial data?*
My hypothesis was that the TCN, which uses parallel dilated convolutions to see the entire history at once, would be inherently more stable than the sequential, memory-dependent LSTM."

---

## Slide 6: Brief Literature Review
**Content:**
- **Zheng et al. (2017):** Established LSTM as the standard (Focus: Accuracy).
- **Bai et al. (2018):** Proposed TCNs as a superior sequence model (Focus: Efficiency).
- **Teh et al. (2020):** Identified "Sensor Integrity" as the missing link.
- **The Gap:** *Lack of robustness testing in RUL literature.*

**Speaker Notes (2:15 - 2:45):**
"Reviewing the literature, Zheng et al. established the LSTM as the gold standard for accuracy in 2017. Later, Bai et al. proposed TCNs, arguing they could match RNNs with better training speed.
However, a critical gap remains. As noted by Teh et al., very few studies stress-test these models. Most research optimizes for accuracy on clean data, ignoring the messy, hostile reality of operational IIoT. My research fills this gap by stress-testing these architectures against cyber-threats."

---

## Slide 7: Methodology
**Visual:**
- **Design:** Quantitative Experimental.
- **Data:** NASA C-MAPSS (Simulated Turbofan Data).
- **Tools:** Python, TensorFlow, Custom 'RUL Analysis Toolkit'.
- **Protocol:**
    1. Train Models.
    2. **Bootstrap Resampling** (n=1000).
    3. **Noise Injection** ($\sigma=0.01$).

**Speaker Notes (2:45 - 3:30):**
"My methodology was purely quantitative. I used the high-fidelity NASA C-MAPSS dataset and built a custom Python toolkit using TensorFlow.
I trained four distinct models. But crucially, I didn't stop at standard testing.
I employed **Bootstrap Resampling** with 1,000 iterations to generate robust confidence intervals, ensuring my results weren't statistical flukes.
Then, I subjected the models to 'Gaussian Noise Injection', deliberately corrupting the test signals to simulate a cyber-attack, and measured the performance drop."

---

## Slide 8: Key Findings of Study
**Visual:**
- **Chart:** TCN vs LSTM Performance (Lower is better).
- **Finding 1:** TCN (RMSE 4.03) > LSTM (RMSE 7.57).
- **Finding 2:** TCN was **Robust** (-0.7% degradation under noise).
- **Finding 3:** Transformers were accurate but **Sensitive** (+0.2% degradation).

**Speaker Notes (3:30 - 4:15):**
"My findings confirmed my hypothesis.
First, the TCN statistically outperformed the LSTM, achieving an error of just 4.03 cycles compared to 7.57.
But the most significant discovery was in specific resilience. Under noise injection, the Transformer and LSTM lost accuracy. The TCN, surprisingly, demonstrated **negative degradation**—it actually handled the noise better than the clean data (-0.7% change).
This suggests the TCN acts as a natural noise filter, making it uniquely suited for secure, hostile environments."

---

## Slide 9: Conclusion
**Visual:**
- **Takeaway:** "Security must be a metric."
- **Implication:** AI Certification needs 'Robustness Stress-Tests'.
- **Future Work:** Test against Gradient-Based (FGSM) attacks.

**Speaker Notes (4:15 - 4:45):**
"To conclude, this study proves that for safety-critical aerospace applications, accuracy is not enough.
The Temporal Convolutional Network offers the best balance of predictive power and cyber-physical resilience.
For the field, this means we must move beyond simple error metrics. Future certification for AI in aviation must include 'Adversarial Robustness Testing' to ensure our systems are safe not just from mechanical failure, but from digital compromise."

---

## Slide 10: References
**Visual:**
- Bai, S. et al. (2018) *An empirical evaluation of generic convolutional and recurrent networks...*
- Saxena, A. et al. (2008) *Damage propagation modeling for aircraft engine...*
- Teh, D. et al. (2020) *Sensor Data Integrity in Industrial IoT...*
- Zheng, S. et al. (2017) *Long short-term memory network for remaining useful life estimation...*

**Speaker Notes (4:45 - 5:00):**
"These are the key references that shaped my methodology. Thank you for your time, and I am happy to take any questions."
