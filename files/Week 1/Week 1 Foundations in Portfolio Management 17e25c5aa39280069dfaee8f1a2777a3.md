# Week 1: Foundations in Portfolio Management

---

**Goal**: Portfolio risk management by understanding key concepts such as **Value at Risk (VaR)**, **Conditional VaR (CVaR)**, **Sharpe Ratio**, and **portfolio optimization**.

---

## **Key Concepts**

### **1. Portfolio**

- **Portfolio:** Basket of investments, like stocks, bonds, mutual funds, or any combination of financial assets.
- The **main objective** of managing a portfolio is to balance **maximizing returns** with **minimizing risk**.
- **Diversification**:
    - “*not putting all your eggs in one basket*.” 
    By spreading investments across different types of assets or markets, you reduce your overall risk.
    - **Example**: Holding tech stocks (high-risk, high-return) and government bonds (low-risk, stable return) creates a more balanced portfolio.

---

### **2. Risk in Finance**

In finance, **Risk** is all about uncertainty and the potential for losses. 

Main types:

- **Default Risk**:
    - Imagine lending money to a friend, and they fail to pay you back. That’s default risk.
    - In financial terms, it’s when a borrower (like a company) fails to meet debt obligations, such as not paying interest on a bond.
    - **Example**: If a company goes bankrupt and can’t repay its bonds, investors lose money.
- **Volatility**:
    - This measures how much the price of an asset goes up and down over time.
    - **High volatility**: Think of cryptocurrencies like Bitcoin, where prices can swing wildly.
    - **Low volatility**: Assets like government bonds are more predictable and stable.
- **Market Risk**:
    - These are risks you can’t control, like an economic downturn or political instability that affects all assets.

---

### **3. Risk Measures**

### **a. Value at Risk (VaR)**

- **What is it?**
    - VaR answers the question: "What’s the **most I could lose over a certain period**, with a given level of confidence?"
- **Key Idea**:
    - It focuses on the downside. For instance, if you’re 95% confident your portfolio won’t lose more than $2,000 in a day, then $2,000 is your **VaR**.
- **Example**:
    - Portfolio Value: $100,000.
    - 95% daily VaR = $2,000.
    - This means there’s a 5% chance the portfolio could lose **more than $2,000** in one day.

### **b. Conditional Value at Risk (CVaR)**

- **Why do we need it?**
    - VaR gives you the "**cut-off**" loss, but it **doesn’t tell you what happens if things get really bad**. That’s where CVaR steps in.
- **Key Idea**:
    - CVaR calculates the **average loss** in the worst-case scenarios (beyond the VaR threshold).
    - It’s like asking, “If I fall off the cliff, how far will I go?”
- **Why It’s Useful**:
    - It’s better for stress testing and understanding extreme risks.

### **c. Sharpe Ratio**

- **What does it do?**
    - The Sharpe Ratio tells you how much **return you’re getting for each unit of risk**.
    - **Formula**:
    
    $\text{Sharpe Ratio} = \frac{\text{Portfolio Return} - \text{Risk-Free Rate}}{\text{Portfolio Standard Deviation}}$
    
- **Example**:
    - If a portfolio gives you a return of 10%, the risk-free rate is 3%, and the portfolio’s volatility (standard deviation) is 15%:
    
    $\text{Sharpe Ratio} = \frac{10\% - 3\%}{15\%} = 0.47$
    
- **Why It Matters**:
    - A **higher Sharpe Ratio** means better risk-adjusted returns. It’s a great way to compare different portfolios.

---

### **4. Portfolio Optimization**

- **What’s the goal?**
    - Find the best mix of assets to maximize returns for a given level of risk.
- **How does it work?**
    - Based on **Modern Portfolio Theory (MPT)**:
        - MPT aims to build an "efficient frontier" — a curve showing the best risk-return combinations.
        - Portfolios along this curve are optimized based on your risk tolerance.
- **Steps**:
    1. Estimate expected returns and risks for each asset.
    2. Calculate **correlations** (relationships) between assets.
    3. Use algorithms to assign weights to assets, balancing risk and return.

---

## **Example Calculations**

### **Value at Risk (VaR)**:

1. Portfolio value = $100,000.
2. Daily volatility = 2%.
3. Confidence level = 95% (1.65 standard deviations for a normal distribution).

$\text{VaR} = \text{Portfolio Value} \times \text{Volatility} \times \text{Z-Score}

\\
\text{VaR} = 100,000 \times 0.02 \times 1.65 = \$3,300$

### **Sharpe Ratio**:

1. Portfolio return = 10%.
2. Risk-free rate = 3%.
3. Standard deviation = 15%.

$\text{Sharpe Ratio} = \frac{10\% - 3\%}{15\%} = 0.47$

---

## **Questions**

1. **VaR vs. CVaR**:
    - How could combining VaR and CVaR provide a fuller picture of risk?
2. **Real-World Challenges**:
    - VaR assumes asset returns follow a normal distribution, but markets often experience "fat tails" (extreme events). How does this limitation affect its reliability?
3. **Correlation and Risk**:
    - Diversification benefits rely on low correlations. How do correlations behave during financial crises, and what does that mean for portfolio optimization?
4. **Stress Testing with GANs**:
    - How could GANs generate synthetic scenarios for market crashes or rare events to improve stress testing?
5. **Sharpe Ratio Critique**:
    - Is a portfolio with a high Sharpe Ratio always safe? What about assets with hidden risks, like those prone to defaults?

---