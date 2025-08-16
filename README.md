# 📈 Dynamic Promotion Personalization using a Contextual Bandit

This project demonstrates a sophisticated A/B testing framework where a machine learning model, specifically a Contextual Multi-Armed Bandit, learns to personalize promotional offers for individual users to maximize conversion rates.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://promotionpersonalization-nikitak.streamlit.app/)

### 🚀 Live Demo

**Check out the interactive dashboard deployed on Streamlit Cloud:**
[https://promotionpersonalization-nikitak.streamlit.app/](https://promotionpersonalization-nikitak.streamlit.app/)

---

### 🎯 Project Overview

In traditional marketing, promotional campaigns often use a one-size-fits-all approach (e.g., "15% off for everyone"). This is suboptimal, as different users are motivated by different types of offers.

This project simulates a real-world scenario where we compare a static control group against a treatment group managed by a reinforcement learning agent. The agent, a **LinUCB (Linear Upper Confidence Bound) contextual bandit**, analyzes each user's profile (their "context") to make a smart decision about which promotion to send them. By balancing **exploration** (trying new offers) and **exploitation** (using the best-known offers), the bandit continuously learns and optimizes its strategy, leading to a significant increase in overall conversions.

### ✨ Key Features

* **Reinforcement Learning:** Implements a Contextual Multi-Armed Bandit (LinUCB) from scratch to make real-time, personalized decisions.
* **A/B Testing Simulation:** A robust simulation framework to compare the bandit's performance against a standard control group.
* **Data Simulation:** Generates a realistic dataset of users with distinct, hidden preferences for the model to learn.
* **Interactive Dashboard:** A web application built with Streamlit to visualize the A/B test results, key performance metrics, and the bandit's learned strategy.

---

###  🛠️ Tech Stack

* **Backend & Modeling:** Python, NumPy, Pandas, Scikit-learn
* **Dashboard:** Streamlit
* **Visualization:** Plotly

---

### 📊 Results & Analysis

The simulation was run with a user base of 5,000 individuals. The bandit model demonstrated a remarkable ability to learn and adapt, resulting in a significant performance lift over the static control strategy.

#### **Key Metrics**

| Metric          | Control Group | Bandit Group | Performance Lift |
| :-------------- | :-----------: | :----------: | :--------------: |
| **Conversion Rate** | 15.60%        | 36.30%       | **+132.69%** |

#### **Performance Over Time**

The cumulative conversion graph clearly shows the bandit's strategy (top line) consistently outperforming the control group. The widening gap between the two lines illustrates the model's effective learning process over time.

![Overall Performance Chart](https://github.com/nikitakumari2/promotion_personalization/blob/main/Overall%20Performance.png)

#### **Bandit's Learned Strategy**

This chart reveals the promotion types the bandit chose to use. It learned that for this particular simulated population, the **"buy_one_get_one"** offer was overwhelmingly the most effective at driving conversions. However, it still allocated a portion of its budget to other offers, demonstrating its ability to personalize and explore options for different user segments.

![Bandit Strategy Chart](https://github.com/nikitakumari2/promotion_personalization/blob/main/Bandit's%20Promotion%20Stratergy.png)

---

### ⚙️ Setup and Installation

To run this project locally, please follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 🚀 Usage

1.  **Run the data simulation script (optional):**
    ```bash
    python simulate_data.py
    ```

2.  **Run the A/B test simulation (optional):**
    ```bash
    python run_simulation.py
    ```

3.  **Launch the Streamlit dashboard:**
    ```bash
    streamlit run app.py
    ```
