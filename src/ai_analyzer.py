import openai
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import time
from pathlib import Path

@dataclass
class AnalysisResult:
    """Structure for AI analysis results."""
    executive_summary: str
    key_insights: List[str]
    critical_issues: List[str]
    recommendations: List[str]
    risk_assessment: str
    confidence_score: float
    analysis_type: str

class AIAnalyzer:
    """
    AI-powered data analysis using GPT-4.
    Handles prompt engineering, response parsing, and cost optimization.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        """
        Initialize the AI analyzer.
        
        Args:
            api_key: OpenAI API key
            model: GPT model to use
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.logger = logging.getLogger(__name__)
        
        # Cost optimization: cache responses
        self.cache = {}
        self.max_cache_size = 100
        
        # Load prompt templates
        self.prompts = self._load_prompt_templates()
    
    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load prompt templates for different analysis types."""
        return {
            'executive': """
You are a senior business analyst presenting to C-level executives. Analyze the provided dataset and generate insights that matter for strategic decision-making.

DATASET CONTEXT:
- Total Rows: {total_rows}
- Columns: {columns}
- Data Quality Score: {quality_score}/100
- Primary Data Types: {data_types}

KEY METRICS:
{key_metrics}

STATISTICAL INSIGHTS:
{statistical_insights}

ANOMALIES DETECTED:
{anomalies}

Please provide a comprehensive executive analysis with the following structure:

🎯 EXECUTIVE SUMMARY (2-3 sentences)
Provide the most important takeaway for executives.

🔍 KEY INSIGHTS (3-5 bullet points)
• Focus on quantified business impact
• Highlight trends and patterns
• Include performance metrics

⚠️ CRITICAL ISSUES (2-4 bullet points)
• Identify risks and problems
• Quantify impact where possible
• Highlight urgent attention areas

📈 STRATEGIC RECOMMENDATIONS (3-5 bullet points)
1. Prioritized action items
2. Resource allocation suggestions
3. Timeline considerations

🎲 RISK ASSESSMENT: [Low/Medium/High]
Brief explanation of overall risk level.

Use clear, business-focused language. Include specific numbers and percentages. Avoid technical jargon.
""",
            
            'financial': """
You are a financial analyst reviewing performance data. Focus on financial metrics, trends, and business impact.

DATASET OVERVIEW:
- Records: {total_rows}
- Analysis Period: {time_period}
- Data Quality: {quality_score}/100

FINANCIAL METRICS:
{financial_metrics}

TRENDS ANALYSIS:
{trend_analysis}

Please provide a financial analysis with:

💰 FINANCIAL PERFORMANCE SUMMARY
Key financial highlights and lowlights.

📊 TREND ANALYSIS
Revenue, cost, and margin trends with specific percentages.

🚨 FINANCIAL RISKS
Areas of concern requiring immediate attention.

💡 OPTIMIZATION OPPORTUNITIES
Specific recommendations for financial improvement.

Focus on ROI, margins, cash flow, and growth metrics.
""",
            
            'operational': """
You are an operations analyst examining performance and efficiency data.

OPERATIONAL CONTEXT:
- Data Points: {total_rows}
- Metrics Tracked: {metrics_count}
- Quality Score: {quality_score}/100

PERFORMANCE INDICATORS:
{performance_data}

EFFICIENCY METRICS:
{efficiency_data}

Provide operational analysis:

⚡ OPERATIONAL EFFICIENCY
Current performance against benchmarks.

🔧 PROCESS INSIGHTS
Bottlenecks and improvement opportunities.

📈 PERFORMANCE TRENDS
Key operational metrics over time.

🎯 ACTION ITEMS
Specific operational improvements ranked by impact.

Focus on productivity, quality, and resource utilization.
""",
            
            'generic': """
You are a data analyst providing comprehensive insights on a business dataset.

DATASET SUMMARY:
- Total Records: {total_rows}
- Data Columns: {column_count}
- Data Quality: {quality_score}/100

DATA CHARACTERISTICS:
{data_summary}

ANALYSIS FINDINGS:
{analysis_findings}

Provide analysis covering:

📊 DATA OVERVIEW
What the data represents and its business context.

🔍 KEY PATTERNS
Important trends and relationships in the data.

⚠️ NOTABLE FINDINGS
Unusual patterns, outliers, or concerning trends.

💡 BUSINESS RECOMMENDATIONS
Actionable insights based on the data analysis.

Make insights relevant to business decision-making.
"""
        }
    
    def analyze_data(self, data_summary: Dict[str, Any], analysis_type: str = 'executive') -> AnalysisResult:
        """
        Perform AI analysis on the data summary.
        
        Args:
            data_summary: Processed data summary from DataProcessor
            analysis_type: Type of analysis ('executive', 'financial', 'operational', 'generic')
            
        Returns:
            AnalysisResult with structured insights
        """
        try:
            # Check cache first
            cache_key = self._generate_cache_key(data_summary, analysis_type)
            if cache_key in self.cache:
                self.logger.info("Using cached analysis result")
                return self.cache[cache_key]
            
            # Prepare the prompt
            prompt = self._prepare_prompt(data_summary, analysis_type)
            
            # Call OpenAI API
            response = self._call_openai_api(prompt)
            
            # Parse response
            result = self._parse_response(response, analysis_type)
            
            # Cache result
            self._cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in AI analysis: {str(e)}")
            return self._create_error_result(str(e))
    
    def _prepare_prompt(self, data_summary: Dict[str, Any], analysis_type: str) -> str:
        """Prepare the prompt for AI analysis."""
        template = self.prompts.get(analysis_type, self.prompts['generic'])
        
        # Extract key information
        insights = data_summary.get('insights', {})
        metadata = data_summary.get('metadata', {})
        
        # Format prompt variables
        prompt_vars = {
            'total_rows': insights.get('dataset_summary', {}).get('total_rows', 'Unknown'),
            'columns': len(metadata.get('columns', [])),
            'column_count': len(metadata.get('columns', [])),
            'quality_score': insights.get('dataset_summary', {}).get('data_quality_score', 'Unknown'),
            'data_types': self._format_data_types(insights.get('dataset_summary', {}).get('primary_data_types', {})),
            'key_metrics': self._format_key_metrics(insights),
            'statistical_insights': self._format_statistical_insights(insights),
            'anomalies': self._format_anomalies(insights.get('anomalies', {})),
            'financial_metrics': self._format_financial_metrics(insights),
            'trend_analysis': self._format_trend_analysis(insights),
            'time_period': self._extract_time_period(insights),
            'metrics_count': len(insights.get('numeric_insights', {})),
            'performance_data': self._format_performance_data(insights),
            'efficiency_data': self._format_efficiency_data(insights),
            'data_summary': self._format_data_summary(data_summary),
            'analysis_findings': self._format_analysis_findings(insights)
        }
        
        return template.format(**prompt_vars)
    
    def _format_data_types(self, data_types: Dict[str, int]) -> str:
        """Format data types for prompt."""
        if not data_types:
            return "Not analyzed"
        
        formatted = []
        for dtype, count in data_types.items():
            if count > 0:
                formatted.append(f"{dtype}: {count}")
        
        return ", ".join(formatted) if formatted else "No data types identified"
    
    def _format_key_metrics(self, insights: Dict[str, Any]) -> str:
        """Format key metrics for prompt."""
        numeric_insights = insights.get('numeric_insights', {})
        
        if not numeric_insights:
            return "No numeric metrics available"
        
        formatted = []
        for col, stats in numeric_insights.items():
            if isinstance(stats, dict):
                mean_val = stats.get('mean', 0)
                formatted.append(f"• {col}: Mean = {mean_val:.2f}, Trend = {stats.get('trend', 'unknown')}")
        
        return "\n".join(formatted[:5])  # Limit to 5 metrics
    
    def _format_statistical_insights(self, insights: Dict[str, Any]) -> str:
        """Format statistical insights for prompt."""
        numeric_insights = insights.get('numeric_insights', {})
        categorical_insights = insights.get('categorical_insights', {})
        
        formatted = []
        
        # Numeric insights
        if numeric_insights:
            formatted.append("NUMERIC DATA:")
            for col, stats in list(numeric_insights.items())[:3]:  # Limit to 3
                if isinstance(stats, dict):
                    outliers = stats.get('outliers', 0)
                    formatted.append(f"• {col}: {outliers} outliers detected")
        
        # Categorical insights
        if categorical_insights:
            formatted.append("CATEGORICAL DATA:")
            for col, info in list(categorical_insights.items())[:3]:  # Limit to 3
                if isinstance(info, dict):
                    unique_vals = info.get('unique_values', 0)
                    formatted.append(f"• {col}: {unique_vals} unique categories")
        
        return "\n".join(formatted) if formatted else "No statistical insights available"
    
    def _format_anomalies(self, anomalies: Dict[str, Any]) -> str:
        """Format anomalies for prompt."""
        if not anomalies:
            return "No anomalies detected"
        
        formatted = []
        
        # Statistical outliers
        outliers = anomalies.get('statistical_outliers', {})
        if outliers:
            formatted.append("STATISTICAL OUTLIERS:")
            for col, info in outliers.items():
                if isinstance(info, dict):
                    count = info.get('count', 0)
                    pct = info.get('percentage', 0)
                    formatted.append(f"• {col}: {count} outliers ({pct:.1f}%)")
        
        # Data quality issues
        quality_issues = anomalies.get('data_quality_issues', {})
        if quality_issues:
            formatted.append("DATA QUALITY ISSUES:")
            for col, issue in quality_issues.items():
                formatted.append(f"• {col}: {issue}")
        
        return "\n".join(formatted) if formatted else "No significant anomalies detected"
    
    def _format_financial_metrics(self, insights: Dict[str, Any]) -> str:
        """Format financial-specific metrics."""
        # Look for financial keywords in column names
        numeric_insights = insights.get('numeric_insights', {})
        financial_keywords = ['revenue', 'cost', 'profit', 'margin', 'price', 'sales', 'expense']
        
        financial_metrics = []
        for col, stats in numeric_insights.items():
            if any(keyword in col.lower() for keyword in financial_keywords):
                if isinstance(stats, dict):
                    mean_val = stats.get('mean', 0)
                    trend = stats.get('trend', 'unknown')
                    financial_metrics.append(f"• {col}: Average = {mean_val:.2f}, Trend = {trend}")
        
        return "\n".join(financial_metrics) if financial_metrics else "No financial metrics identified"
    
    def _format_trend_analysis(self, insights: Dict[str, Any]) -> str:
        """Format trend analysis."""
        numeric_insights = insights.get('numeric_insights', {})
        
        trends = {'increasing': [], 'decreasing': [], 'stable': []}
        
        for col, stats in numeric_insights.items():
            if isinstance(stats, dict):
                trend = stats.get('trend', 'unknown')
                if trend in trends:
                    trends[trend].append(col)
        
        formatted = []
        for trend_type, columns in trends.items():
            if columns:
                formatted.append(f"{trend_type.upper()}: {', '.join(columns)}")
        
        return "\n".join(formatted) if formatted else "No trend analysis available"
    
    def _extract_time_period(self, insights: Dict[str, Any]) -> str:
        """Extract time period from temporal insights."""
        temporal_insights = insights.get('temporal_insights', {})
        
        if not temporal_insights:
            return "Time period not determined"
        
        periods = []
        for col, info in temporal_insights.items():
            if isinstance(info, dict):
                span = info.get('time_span_days', 0)
                if span > 0:
                    if span < 7:
                        periods.append(f"{span} days")
                    elif span < 30:
                        periods.append(f"{span//7} weeks")
                    elif span < 365:
                        periods.append(f"{span//30} months")
                    else:
                        periods.append(f"{span//365} years")
        
        return ", ".join(periods) if periods else "Time period not determined"
    
    def _format_performance_data(self, insights: Dict[str, Any]) -> str:
        """Format performance data for operational analysis."""
        return self._format_key_metrics(insights)
    
    def _format_efficiency_data(self, insights: Dict[str, Any]) -> str:
        """Format efficiency data for operational analysis."""
        return self._format_statistical_insights(insights)
    
    def _format_data_summary(self, data_summary: Dict[str, Any]) -> str:
        """Format general data summary."""
        metadata = data_summary.get('metadata', {})
        insights = data_summary.get('insights', {})
        
        summary_parts = []
        
        # Basic info
        shape = metadata.get('shape', [0, 0])
        summary_parts.append(f"Dataset: {shape[0]} rows × {shape[1]} columns")
        
        # Data quality
        quality_score = insights.get('dataset_summary', {}).get('data_quality_score', 0)
        summary_parts.append(f"Data Quality Score: {quality_score}/100")
        
        # Column types
        column_types = insights.get('dataset_summary', {}).get('primary_data_types', {})
        if column_types:
            types_str = ", ".join([f"{k}: {v}" for k, v in column_types.items() if v > 0])
            summary_parts.append(f"Column Types: {types_str}")
        
        return "\n".join(summary_parts)
    
    def _format_analysis_findings(self, insights: Dict[str, Any]) -> str:
        """Format general analysis findings."""
        findings = []
        
        # Numeric findings
        numeric_insights = insights.get('numeric_insights', {})
        if numeric_insights:
            findings.append(f"Analyzed {len(numeric_insights)} numeric columns")
            
            # Count trends
            trends = {}
            for stats in numeric_insights.values():
                if isinstance(stats, dict):
                    trend = stats.get('trend', 'unknown')
                    trends[trend] = trends.get(trend, 0) + 1
            
            if trends:
                trend_summary = ", ".join([f"{count} {trend}" for trend, count in trends.items()])
                findings.append(f"Trends: {trend_summary}")
        
        # Categorical findings
        categorical_insights = insights.get('categorical_insights', {})
        if categorical_insights:
            findings.append(f"Analyzed {len(categorical_insights)} categorical columns")
        
        # Anomalies
        anomalies = insights.get('anomalies', {})
        outlier_count = len(anomalies.get('statistical_outliers', {}))
        quality_issues = len(anomalies.get('data_quality_issues', {}))
        if outlier_count > 0 or quality_issues > 0:
            findings.append(f"Issues found: {outlier_count} columns with outliers, {quality_issues} data quality issues")
        
        return "\n".join(findings) if findings else "Basic analysis completed"
    
    def _call_openai_api(self, prompt: str) -> str:
        """Call OpenAI API with error handling and retry logic."""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are an expert business analyst. Provide clear, actionable insights based on data analysis. Use business-friendly language and focus on strategic implications."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1500,
                    temperature=0.3,  # Lower temperature for more consistent analysis
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                self.logger.warning(f"API call attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise e
    
    def _parse_response(self, response: str, analysis_type: str) -> AnalysisResult:
        """Parse AI response into structured result."""
        try:
            # Initialize with defaults
            executive_summary = ""
            key_insights = []
            critical_issues = []
            recommendations = []
            risk_assessment = "Medium"
            confidence_score = 0.8
            
            # Parse different sections based on emojis and headers
            lines = response.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Identify sections
                if '🎯' in line or 'EXECUTIVE SUMMARY' in line.upper():
                    current_section = 'executive'
                    continue
                elif '🔍' in line or 'KEY INSIGHTS' in line.upper():
                    current_section = 'insights'
                    continue
                elif '⚠️' in line or 'CRITICAL ISSUES' in line.upper():
                    current_section = 'issues'
                    continue
                elif '📈' in line or 'RECOMMENDATIONS' in line.upper():
                    current_section = 'recommendations'
                    continue
                elif '🎲' in line or 'RISK ASSESSMENT' in line.upper():
                    current_section = 'risk'
                    # Extract risk level from this line
                    if 'Low' in line:
                        risk_assessment = "Low"
                    elif 'High' in line:
                        risk_assessment = "High"
                    else:
                        risk_assessment = "Medium"
                    continue
                
                # Add content to appropriate sections
                if current_section == 'executive' and len(line) > 10:
                    executive_summary += line + " "
                elif current_section == 'insights' and line.startswith(('•', '-', '1.', '2.', '3.', '4.', '5.')):
                    key_insights.append(line.lstrip('•- 1234567890.').strip())
                elif current_section == 'issues' and line.startswith(('•', '-', '1.', '2.', '3.', '4.', '5.')):
                    critical_issues.append(line.lstrip('•- 1234567890.').strip())
                elif current_section == 'recommendations' and line.startswith(('•', '-', '1.', '2.', '3.', '4.', '5.')):
                    recommendations.append(line.lstrip('•- 1234567890.').strip())
            
            # Clean up executive summary
            executive_summary = executive_summary.strip()
            
            # Calculate confidence based on content quality
            confidence_score = self._calculate_confidence(
                executive_summary, key_insights, critical_issues, recommendations
            )
            
            return AnalysisResult(
                executive_summary=executive_summary or "Analysis completed successfully.",
                key_insights=key_insights or ["Data analysis performed on uploaded dataset"],
                critical_issues=critical_issues or ["No critical issues identified"],
                recommendations=recommendations or ["Continue monitoring data quality"],
                risk_assessment=risk_assessment,
                confidence_score=confidence_score,
                analysis_type=analysis_type
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing AI response: {str(e)}")
            return self._create_error_result(f"Error parsing response: {str(e)}")
    
    def _calculate_confidence(self, summary: str, insights: List[str], issues: List[str], recommendations: List[str]) -> float:
        """Calculate confidence score based on response quality."""
        score = 0.5  # Base score
        
        # Add points for content quality
        if len(summary) > 50:
            score += 0.1
        if len(insights) >= 3:
            score += 0.1
        if len(issues) >= 1:
            score += 0.1
        if len(recommendations) >= 3:
            score += 0.1
        
        # Add points for specific business terms
        business_terms = ['revenue', 'cost', 'profit', 'efficiency', 'performance', 'growth', 'trend']
        full_text = f"{summary} {' '.join(insights)} {' '.join(recommendations)}".lower()
        
        term_count = sum(1 for term in business_terms if term in full_text)
        score += min(0.1, term_count * 0.02)
        
        return min(1.0, score)
    
    def _generate_cache_key(self, data_summary: Dict[str, Any], analysis_type: str) -> str:
        """Generate cache key for storing results."""
        # Create a simple hash of key data characteristics
        insights = data_summary.get('insights', {})
        dataset_summary = insights.get('dataset_summary', {})
        
        key_elements = [
            str(dataset_summary.get('total_rows', 0)),
            str(dataset_summary.get('total_columns', 0)),
            str(dataset_summary.get('data_quality_score', 0)),
            analysis_type,
            str(len(insights.get('numeric_insights', {}))),
            str(len(insights.get('categorical_insights', {})))
        ]
        
        return "_".join(key_elements)
    
    def _cache_result(self, cache_key: str, result: AnalysisResult):
        """Cache analysis result."""
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[cache_key] = result
    
    def _create_error_result(self, error_message: str) -> AnalysisResult:
        """Create error result for failed analysis."""
        return AnalysisResult(
            executive_summary=f"Analysis failed: {error_message}",
            key_insights=["Unable to generate insights due to error"],
            critical_issues=[f"Error in analysis: {error_message}"],
            recommendations=["Please check data format and try again"],
            risk_assessment="Unknown",
            confidence_score=0.0,
            analysis_type="error"
        )
    
    def get_analysis_types(self) -> List[str]:
        """Get available analysis types."""
        return list(self.prompts.keys())
    
    def format_result_for_display(self, result: AnalysisResult) -> str:
        """Format analysis result for display."""
        formatted = f"""
# 📊 {result.analysis_type.title()} Analysis Report

## 🎯 Executive Summary
{result.executive_summary}

## 🔍 Key Insights
"""
        
        for i, insight in enumerate(result.key_insights, 1):
            formatted += f"{i}. {insight}\n"
        
        formatted += "\n## ⚠️ Critical Issues\n"
        for i, issue in enumerate(result.critical_issues, 1):
            formatted += f"{i}. {issue}\n"
        
        formatted += "\n## 📈 Recommendations\n"
        for i, rec in enumerate(result.recommendations, 1):
            formatted += f"{i}. {rec}\n"
        
        formatted += f"""
## 🎲 Risk Assessment
**Level:** {result.risk_assessment}

## 📋 Analysis Metadata
- **Confidence Score:** {result.confidence_score:.1%}
- **Analysis Type:** {result.analysis_type}
"""
        
        return formatted