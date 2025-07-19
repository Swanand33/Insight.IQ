# 🧠 Insight.IQ : AI-Powered Excel Analytics

> Transform spreadsheets into executive insights in seconds using GPT-4

## 💡 The Problem

Business analysts spend **60%+ of their time** wrestling with Excel instead of generating insights. Most reports sit unread because extracting key findings takes too long and lacks visual impact.

## ⚡ The Solution

InsightGenie automatically transforms your Excel/CSV files into:

- 📊 **Executive Summaries** - AI-powered business insights in plain English
- 🎯 **Strategic Recommendations** - Actionable next steps based on data patterns  
- 📈 **Interactive Dashboards** - Auto-generated charts and visualizations
- 🔍 **Anomaly Detection** - Identify outliers and data quality issues
- 📱 **Export-Ready Reports** - Professional markdown and HTML outputs

---

## 📋 Project Structure

```
insightgenie/
├── 📄 README.md                    # Project documentation
├── 📄 requirements.txt             # Python dependencies
├── 📄 Dockerfile                   # Container deployment
├── 📁 src/                         # Core application
│   ├── main.py                     # Gradio interface
│   ├── data_processor.py           # Data analysis engine
│   ├── ai_analyzer.py              # GPT-4 integration
│   ├── visualization.py            # Chart generation
│   └── utils.py                    # Helper functions
├── 📁 prompts/                     # AI analysis templates
├── 📁 tests/                       # Test suite
├── 📁 docs/                        # Documentation
└── 📁 examples/                    # Sample data
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/yourusername/insightgenie
cd insightgenie
pip install -r requirements.txt
python src/main.py
```

Open `http://localhost:7860` → Upload Excel/CSV → Get AI insights!

## ✨ What It Does

- **📊 Data Analysis**: Automatic quality assessment and insights
- **🤖 AI Reports**: GPT-4 generates executive summaries  
- **📈 Visualizations**: Auto-generated interactive charts
- **📱 Export**: Download reports and dashboards

## 🎯 Example Output

```
🎯 EXECUTIVE SUMMARY
Q3 revenue increased 23% QoQ driven by Enterprise growth.

🔍 KEY INSIGHTS
• Revenue: $4.2M (+18% vs Q2, +34% YoY)
• Southwest region: $2.1M above forecast

⚠️ CRITICAL ISSUES  
• Customer churn increased 15% in August

📈 RECOMMENDATIONS
1. Expand Enterprise sales team
2. Implement retention program
```

## 🛠️ Tech Stack

- **Frontend**: Gradio (Python web UI)
- **Data**: Pandas + NumPy  
- **AI**: OpenAI GPT-4
- **Charts**: Plotly + Matplotlib
- **Files**: Excel, CSV support


## 🎓 Skills Demonstrated

✅ **AI/ML Engineering** - GPT-4 integration, prompt engineering  
✅ **Data Science** - Pandas, statistical analysis, visualization  
✅ **Web Development** - Gradio interface, responsive design  
✅ **Software Engineering** - Testing, documentation, deployment  

## 🔧 Features

- Upload Excel/CSV files (up to 50MB)
- 4 analysis types: Executive, Financial, Operational, Generic
- Auto-generated charts and dashboards
- Professional report exports
- Real-time processing

## 📊 Performance

- **50MB Excel**: <20 seconds
- **100K rows**: <30 seconds  
- **AI Analysis**: 10-15 seconds
- **Charts**: <15 seconds

## 🚀 Deployment

```bash
# Local
python src/main.py

# Docker
docker build -t insightgenie .
docker run -p 7860:7860 insightgenie

# Cloud
gradio deploy src/main.py
```

## 🔑 Setup

```bash
# Optional: Add OpenAI API key for AI features
export OPENAI_API_KEY="sk-your-key-here"

# Run without API key for data processing only
python src/main.py
```

## 📋 Use Cases

- **Executive Reporting**: Monthly board presentations
- **Sales Analysis**: Territory performance insights
- **HR Analytics**: Employee satisfaction trends
- **Operations**: Process efficiency optimization

---