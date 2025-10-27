# 🧠 Market Mind - A Live Market Regime Dashboard

**Market Mind** is an interactive **Streamlit web dashboard** that tracks and visualizes the *current market mood* — identifying **Risk-On / Risk-Off regimes** using S&P 500 price data, volatility, and breadth signals.

It provides a **composite market score**, dynamic visualizations, and interactive controls to help users analyze market behavior and detect trend shifts.

🌐 **Live Demo:** https://market-mind-c8rpuazxg3lwpu6cbnsn2t.streamlit.app/
---

## 🚀 Features

- 📊 **Composite Market Score**
  - Combines **momentum**, **volatility**, **SPX–VIX correlation**, and **breadth** indicators.
- 🎯 **Market Regime Classification**
  - Visualizes **Risk-On / Neutral / Risk-Off** phases dynamically.
- ⚙️ **Adjustable Signal Parameters**
  - Tune window lengths, weightings, and lookback periods live.
- 🧮 **Sector & Strategy Tabs**
  - Analyze sector rotation and regime-based backtesting.
- 🗓️ **Live Data via Yahoo Finance**
  - Fetches the latest S&P 500 and VIX data automatically.
- 🧠 **Intuitive Visualization**
  - Color-coded regime bands for instant interpretation.

---

## 💡 Methodology

The dashboard calculates a **Market Score**:
```text
Market Score = w₁(Momentum) + w₂(-SPX Volatility) + w₃(-VIX Volatility) + w₄(-Corr(SPX, VIX)) + w₅(Breadth)

Market Mind is part of a larger framework of data-driven systems I’m building, connecting finance, behavior, and systems design. From markets to orbital habitats, it’s all about modeling how systems behave under uncertainty.



