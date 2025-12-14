# Customer Churn Analyzer — Dark PowerBI + RAG

**PowerBI-style dark dashboard with interactive churn analytics and Perplexity Sonar RAG Q&A. Upload CSV → Visualize → Ask questions about your data.**

## ✨ Features

- **📊 PowerBI Dark Theme** — KPI metrics, retention curves, boxplots, heatmaps
- **🎯 Churn Analysis** — By contract, payment, services, tenure survival
- **🔎 RAG Q&A** — TF-IDF retrieval + Perplexity Sonar LLM answers
- **⚙️ Filters** — Segment by Contract, PaymentMethod, InternetService, etc.
- **💾 Export** — Download filtered dataset

## 🚀 Quick Start

pip install -r requirements.txt
streamlit run app.py


**Streamlit Secrets** (`.streamlit/secrets.toml`):
PERPLEXITY_API_KEY = "pplx-XXXXXXXXXXXXXXXX"

## 📁 Upload CSV Format

## 🛠️ Requirements.txt
streamlit>=1.24.0
pandas>=2.2.3
numpy>=2.1.0
plotly>=5.13.1
scikit-learn>=1.2.2
openai>=1.0.0


## 📱 Demo

Upload any Telco-style CSV with columns like:
- `customerID`, `tenure`, `MonthlyCharges`
- `Contract`, `InternetService`, `PaymentMethod`
- `Churn` (Yes/No or 1/0)

**Configure columns in sidebar → Run Analysis → Ask RAG questions!**

## 🔧 Usage

1. **Upload CSV** or use sample data
2. **Sidebar**: Map Churn, Tenure columns + apply filters
3. **🚀 Run Analysis** → See PowerBI-style dashboard
4. **🔎 RAG Q&A**: Ask "Which contract has highest churn?" etc.

## 💡 RAG Questions Examples
- "Which contract has highest churn?"
- "Show customers with high charges who churned"
- "What payment method is riskiest?"

## 🎨 Screenshots
*(Add your screenshots here)*

## 📄 License
MIT License

---

**⭐ Star if useful!** Built for customer churn analysis with production-ready visuals + AI insights.
