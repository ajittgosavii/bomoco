"""
BOMOCO Claude AI Integration
Intelligent cloud optimization powered by Anthropic Claude

Features:
- Natural language query interface
- AI-powered recommendation explanations
- Executive report generation
- Anomaly detection and insights
- Conversational cloud assistant
"""

import anthropic
import json
import streamlit as st
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import pandas as pd


@dataclass
class AIInsight:
    """Represents an AI-generated insight."""
    category: str
    title: str
    content: str
    confidence: float
    action_items: List[str]
    timestamp: datetime


class ClaudeCloudAssistant:
    """
    Claude-powered cloud optimization assistant.
    Provides intelligent analysis and natural language interactions.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Claude assistant."""
        self.api_key = api_key or self._get_api_key()
        self.client = None
        self.model = "claude-sonnet-4-20250514"
        
        if self.api_key:
            try:
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except Exception as e:
                print(f"Failed to initialize Claude client: {e}")
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from Streamlit secrets or environment."""
        try:
            return st.secrets.get("anthropic", {}).get("api_key", "")
        except Exception:
            import os
            return os.getenv("ANTHROPIC_API_KEY", "")
    
    @property
    def is_available(self) -> bool:
        """Check if Claude API is available."""
        return self.client is not None and bool(self.api_key)
    
    def _create_system_prompt(self, context: str = "general") -> str:
        """Create context-specific system prompt."""
        base_prompt = """You are BOMOCO AI, an expert cloud optimization assistant powered by Claude. 
You help users optimize their cloud infrastructure across cost, carbon emissions, water usage, 
performance, and business outcomes.

Your capabilities:
- Analyze cloud spending patterns and identify savings opportunities
- Recommend sustainable infrastructure decisions (carbon, water)
- Explain trade-offs between cost, performance, and sustainability
- Generate executive summaries and reports
- Answer questions about AWS, Azure, and GCP optimization
- Provide actionable, specific recommendations

Guidelines:
- Be concise and actionable
- Use specific numbers and percentages when available
- Prioritize recommendations by impact
- Consider business context when making suggestions
- Highlight quick wins vs. long-term improvements
- Always explain the "why" behind recommendations
"""
        
        context_additions = {
            "cost_analysis": "\nFocus on cost optimization, identifying waste, rightsizing opportunities, and reserved instance recommendations.",
            "sustainability": "\nFocus on carbon footprint reduction, water usage optimization, and sustainable cloud practices.",
            "performance": "\nFocus on performance optimization, latency reduction, and reliability improvements.",
            "executive": "\nProvide high-level summaries suitable for executive leadership. Focus on business impact and ROI.",
            "technical": "\nProvide detailed technical analysis with specific implementation steps and configurations.",
        }
        
        return base_prompt + context_additions.get(context, "")
    
    def chat(
        self,
        message: str,
        context: str = "general",
        workload_data: Optional[pd.DataFrame] = None,
        recommendations: Optional[List] = None,
        conversation_history: Optional[List[Dict]] = None,
    ) -> str:
        """
        Chat with Claude about cloud optimization.
        
        Args:
            message: User's question or request
            context: Context type (general, cost_analysis, sustainability, etc.)
            workload_data: Optional DataFrame with current workload data
            recommendations: Optional list of current recommendations
            conversation_history: Optional previous messages for context
            
        Returns:
            Claude's response
        """
        if not self.is_available:
            return self._get_fallback_response(message)
        
        # Build context from data
        data_context = self._build_data_context(workload_data, recommendations)
        
        # Build messages
        messages = []
        
        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history[-10:])  # Last 10 messages
        
        # Add current message with data context
        user_content = message
        if data_context:
            user_content = f"""Current Infrastructure Data:
{data_context}

User Question: {message}"""
        
        messages.append({"role": "user", "content": user_content})
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=self._create_system_prompt(context),
                messages=messages,
            )
            return response.content[0].text
        except Exception as e:
            return f"I encountered an error: {str(e)}. Please try again."
    
    def _build_data_context(
        self,
        workload_data: Optional[pd.DataFrame],
        recommendations: Optional[List],
    ) -> str:
        """Build context string from available data."""
        parts = []
        
        if workload_data is not None and not workload_data.empty:
            summary = {
                "total_workloads": len(workload_data),
                "total_monthly_cost": f"${workload_data['monthly_cost'].sum():,.0f}",
                "regions": workload_data['region'].nunique(),
                "avg_cpu_utilization": f"{workload_data['cpu_utilization'].mean()*100:.1f}%",
                "workload_types": workload_data['workload_type'].value_counts().to_dict(),
            }
            parts.append(f"Workload Summary: {json.dumps(summary, indent=2)}")
        
        if recommendations:
            rec_summary = [
                {
                    "action": r.action_type,
                    "workload": r.workload_id,
                    "savings": f"${r.estimated_savings_monthly:,.0f}/mo",
                    "carbon_impact": f"{r.carbon_impact:+.1f}%",
                    "confidence": f"{r.confidence:.0%}",
                }
                for r in recommendations[:5]  # Top 5
            ]
            parts.append(f"Top Recommendations: {json.dumps(rec_summary, indent=2)}")
        
        return "\n\n".join(parts)
    
    def _get_fallback_response(self, message: str) -> str:
        """Provide helpful response when Claude API is unavailable."""
        return """🤖 **Claude AI is not configured yet.**

To enable AI-powered insights:

1. Get an API key from [console.anthropic.com](https://console.anthropic.com)
2. Add to Streamlit secrets:
   ```toml
   [anthropic]
   api_key = "sk-ant-..."
   ```
3. Restart the app

In the meantime, you can still use all the optimization features - just without AI explanations!"""
    
    def analyze_recommendations(
        self,
        recommendations: List,
        workload_data: pd.DataFrame,
        focus: str = "balanced",
    ) -> str:
        """
        Get AI analysis of optimization recommendations.
        
        Args:
            recommendations: List of OptimizationAction objects
            workload_data: Current workload data
            focus: Analysis focus (balanced, cost, sustainability, performance)
            
        Returns:
            Detailed AI analysis
        """
        if not self.is_available:
            return self._get_fallback_response("analyze")
        
        prompt = f"""Analyze these cloud optimization recommendations and provide insights:

Focus Area: {focus}

Please provide:
1. **Executive Summary** (2-3 sentences)
2. **Top 3 Priority Actions** with reasoning
3. **Risk Assessment** for the recommendations
4. **Expected Outcomes** (cost savings, carbon reduction, performance impact)
5. **Implementation Roadmap** (quick wins vs. longer-term changes)

Be specific with numbers and actionable advice."""
        
        return self.chat(
            prompt,
            context="technical" if focus == "performance" else "cost_analysis",
            workload_data=workload_data,
            recommendations=recommendations,
        )
    
    def generate_executive_report(
        self,
        workload_data: pd.DataFrame,
        recommendations: List,
        sustainability_metrics: Dict,
        period: str = "monthly",
    ) -> str:
        """
        Generate executive summary report.
        
        Args:
            workload_data: Current workload data
            recommendations: Optimization recommendations
            sustainability_metrics: Sustainability metrics dict
            period: Report period (weekly, monthly, quarterly)
            
        Returns:
            Executive report in markdown format
        """
        if not self.is_available:
            return self._get_fallback_response("report")
        
        prompt = f"""Generate a {period} executive report for cloud infrastructure optimization.

Sustainability Metrics:
{json.dumps(sustainability_metrics, indent=2)}

Please create a professional report with:
1. **Executive Summary** - Key highlights and overall health
2. **Financial Overview** - Current spend, savings achieved, opportunities
3. **Sustainability Score** - Carbon footprint, water usage, improvements
4. **Performance Metrics** - Reliability, latency, availability
5. **Key Recommendations** - Top 3 actions for next period
6. **Risk & Compliance** - Any concerns or items needing attention

Format as a clean, professional markdown report suitable for C-level executives."""
        
        return self.chat(
            prompt,
            context="executive",
            workload_data=workload_data,
            recommendations=recommendations,
        )
    
    def explain_recommendation(self, recommendation, workload_data: pd.DataFrame) -> str:
        """
        Get detailed explanation for a specific recommendation.
        
        Args:
            recommendation: Single OptimizationAction object
            workload_data: Current workload data
            
        Returns:
            Detailed explanation
        """
        if not self.is_available:
            return self._get_fallback_response("explain")
        
        prompt = f"""Explain this cloud optimization recommendation in detail:

Action: {recommendation.action_type}
Workload: {recommendation.workload_id}
Description: {recommendation.description}
Cost Impact: {recommendation.cost_impact:+.1f}%
Carbon Impact: {recommendation.carbon_impact:+.1f}%
Estimated Savings: ${recommendation.estimated_savings_monthly:,.0f}/month
Risk Level: {recommendation.risk_level}
Confidence: {recommendation.confidence:.0%}

Please explain:
1. **Why this recommendation?** - What triggered this suggestion
2. **How it works** - Technical explanation
3. **Benefits** - Detailed breakdown of expected improvements
4. **Risks & Mitigations** - What could go wrong and how to prevent it
5. **Implementation Steps** - Step-by-step guide
6. **Rollback Plan** - How to reverse if needed

Make it understandable for both technical and non-technical readers."""
        
        return self.chat(prompt, context="technical", workload_data=workload_data)
    
    def detect_anomalies(
        self,
        workload_data: pd.DataFrame,
        historical_context: Optional[str] = None,
    ) -> str:
        """
        Detect anomalies and unusual patterns in infrastructure.
        
        Args:
            workload_data: Current workload data
            historical_context: Optional historical comparison data
            
        Returns:
            Anomaly analysis
        """
        if not self.is_available:
            return self._get_fallback_response("anomalies")
        
        # Calculate basic anomaly indicators
        anomaly_indicators = {
            "low_utilization_workloads": len(workload_data[workload_data['cpu_utilization'] < 0.2]),
            "high_utilization_workloads": len(workload_data[workload_data['cpu_utilization'] > 0.85]),
            "high_cost_low_util": len(workload_data[
                (workload_data['cpu_utilization'] < 0.3) & 
                (workload_data['monthly_cost'] > workload_data['monthly_cost'].median())
            ]),
            "cost_outliers": len(workload_data[
                workload_data['monthly_cost'] > workload_data['monthly_cost'].mean() + 2*workload_data['monthly_cost'].std()
            ]),
        }
        
        prompt = f"""Analyze this infrastructure data for anomalies and unusual patterns:

Anomaly Indicators:
{json.dumps(anomaly_indicators, indent=2)}

{f"Historical Context: {historical_context}" if historical_context else ""}

Please identify:
1. **Critical Anomalies** - Issues requiring immediate attention
2. **Warning Signs** - Potential problems developing
3. **Unusual Patterns** - Deviations from expected behavior
4. **Cost Anomalies** - Unexpected spending patterns
5. **Performance Concerns** - Workloads at risk

For each finding, provide severity (🔴 Critical, 🟡 Warning, 🟢 Info) and recommended action."""
        
        return self.chat(prompt, context="technical", workload_data=workload_data)
    
    def answer_cloud_question(self, question: str) -> str:
        """
        Answer general cloud optimization questions.
        
        Args:
            question: User's question about cloud optimization
            
        Returns:
            Helpful answer
        """
        if not self.is_available:
            return self._get_fallback_response("question")
        
        return self.chat(question, context="general")
    
    def suggest_sustainability_improvements(
        self,
        workload_data: pd.DataFrame,
        current_carbon_kg: float,
        current_water_liters: float,
    ) -> str:
        """
        Get AI suggestions for sustainability improvements.
        
        Args:
            workload_data: Current workload data
            current_carbon_kg: Current monthly carbon footprint
            current_water_liters: Current monthly water usage
            
        Returns:
            Sustainability recommendations
        """
        if not self.is_available:
            return self._get_fallback_response("sustainability")
        
        prompt = f"""Provide sustainability improvement recommendations for this cloud infrastructure:

Current Metrics:
- Monthly Carbon Footprint: {current_carbon_kg:,.0f} kg CO2
- Monthly Water Usage: {current_water_liters:,.0f} liters
- Number of Workloads: {len(workload_data)}

Regions in use: {workload_data['region'].unique().tolist()}

Please provide:
1. **Quick Wins** - Immediate actions for 10-20% reduction
2. **Medium-term Improvements** - 3-6 month initiatives
3. **Strategic Changes** - Long-term sustainability transformation
4. **Region Optimization** - Which workloads to move to greener regions
5. **Scheduling Opportunities** - Workloads that can shift to low-carbon hours
6. **Benchmarking** - How this compares to industry standards

Include specific, actionable recommendations with expected impact."""
        
        return self.chat(prompt, context="sustainability", workload_data=workload_data)


def render_ai_chat_interface(assistant: ClaudeCloudAssistant, workload_data: pd.DataFrame, recommendations: List):
    """Render the AI chat interface in Streamlit."""
    
    st.markdown("### 🤖 Claude AI Assistant")
    
    if not assistant.is_available:
        st.warning("⚠️ Claude AI not configured. Add your API key in Settings → Secrets.")
        st.code("""[anthropic]
api_key = "sk-ant-your-key-here"
""", language="toml")
        return
    
    st.success("✅ Claude AI connected")
    
    # Initialize chat history
    if "ai_messages" not in st.session_state:
        st.session_state.ai_messages = []
    
    # Quick action buttons
    st.markdown("**Quick Actions:**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Analyze Costs", use_container_width=True):
            st.session_state.ai_pending_action = "analyze_costs"
    with col2:
        if st.button("🌱 Sustainability Tips", use_container_width=True):
            st.session_state.ai_pending_action = "sustainability"
    with col3:
        if st.button("📝 Executive Report", use_container_width=True):
            st.session_state.ai_pending_action = "executive_report"
    with col4:
        if st.button("🔍 Find Anomalies", use_container_width=True):
            st.session_state.ai_pending_action = "anomalies"
    
    # Handle quick actions
    if "ai_pending_action" in st.session_state:
        action = st.session_state.ai_pending_action
        del st.session_state.ai_pending_action
        
        with st.spinner("🤖 Claude is analyzing..."):
            if action == "analyze_costs":
                response = assistant.analyze_recommendations(recommendations, workload_data, "cost")
            elif action == "sustainability":
                from data.sample_data import calculate_sustainability_metrics
                metrics = calculate_sustainability_metrics(workload_data)
                response = assistant.suggest_sustainability_improvements(
                    workload_data, metrics['total_monthly_carbon_kg'], metrics['total_monthly_water_liters']
                )
            elif action == "executive_report":
                from data.sample_data import calculate_sustainability_metrics
                metrics = calculate_sustainability_metrics(workload_data)
                response = assistant.generate_executive_report(workload_data, recommendations, metrics)
            elif action == "anomalies":
                response = assistant.detect_anomalies(workload_data)
            else:
                response = "Unknown action"
        
        st.session_state.ai_messages.append({"role": "assistant", "content": response})
    
    # Display chat history
    for msg in st.session_state.ai_messages[-10:]:  # Last 10 messages
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your cloud infrastructure..."):
        st.session_state.ai_messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = assistant.chat(
                    prompt,
                    workload_data=workload_data,
                    recommendations=recommendations,
                    conversation_history=st.session_state.ai_messages[:-1],
                )
            st.markdown(response)
        
        st.session_state.ai_messages.append({"role": "assistant", "content": response})
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.ai_messages = []
        st.rerun()
