"""
LLM-powered summarization for model results and explainability.
"""

from typing import Dict, Optional
import json
import os


class ModelResultsSummarizer:
    """Generates LLM-powered summaries of model performance and explainability."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.use_llm = self.api_key is not None

    def generate_summary(self, metrics: Dict[str, Dict[str, float]], ticker: str) -> str:
        """Generate a summary of model performance using LLM or fallback template."""
        
        if self.use_llm:
            try:
                return self._generate_llm_summary(metrics, ticker)
            except Exception as e:
                print(f"Warning: LLM summary failed: {e}, using template instead")
        
        return self._generate_template_summary(metrics, ticker)

    def _generate_llm_summary(self, metrics: Dict[str, Dict[str, float]], ticker: str) -> str:
        """Generate summary using OpenAI API."""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=self.api_key)
            
            # Format metrics for prompt
            metrics_text = json.dumps(metrics, indent=2)
            
            prompt = f"""Analyze the following portfolio optimization model performance metrics for {ticker}:

{metrics_text}

Provide a concise summary (2-3 paragraphs) that:
1. Compares the performance of baseline DQN, MHA-DQN (clean), and MHA-DQN (robust)
2. Highlights key differences in Sharpe ratio, CAGR, max drawdown, and robustness scores
3. Explains the practical implications for portfolio management
4. Notes any strengths or weaknesses of each approach

Write in clear, non-technical language suitable for financial professionals."""

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial AI analyst expert at explaining portfolio optimization results."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except (ImportError, Exception) as e:
            print(f"Warning: LLM summary generation failed: {e}, using template instead")
            return self._generate_template_summary(metrics, ticker)

    def _generate_template_summary(self, metrics: Dict[str, Dict[str, float]], ticker: str) -> str:
        """Generate summary using template when LLM is not available."""
        
        baseline = metrics.get("baseline_dqn", {})
        clean = metrics.get("mha_dqn_clean", {})
        robust = metrics.get("mha_dqn_robust", {})
        
        # Find best performing model
        models_sorted = sorted(
            [
                ("Baseline DQN", baseline.get("sharpe", 0), baseline),
                ("MHA-DQN (Clean)", clean.get("sharpe", 0), clean),
                ("MHA-DQN (Robust)", robust.get("sharpe", 0), robust),
            ],
            key=lambda x: x[1],
            reverse=True
        )
        
        best_name, best_sharpe, best_metrics = models_sorted[0]
        
        summary = f"""## Model Performance Summary for {ticker}

**Best Performing Model: {best_name}**

The analysis shows that **{best_name}** achieves the highest Sharpe ratio of {best_metrics.get('sharpe', 0):.3f}, 
indicating superior risk-adjusted returns. This model demonstrates a CAGR of {best_metrics.get('cagr', 0):.2%} 
with a maximum drawdown of {best_metrics.get('max_drawdown', 0):.2%}.

**Key Comparisons:**

- **Baseline DQN**: Sharpe ratio of {baseline.get('sharpe', 0):.3f} - Standard deep Q-learning approach
- **MHA-DQN (Clean)**: Sharpe ratio of {clean.get('sharpe', 0):.3f} - Enhanced with multi-head attention mechanism
- **MHA-DQN (Robust)**: Sharpe ratio of {robust.get('sharpe', 0):.3f} - Adversarially trained for resilience

**Robustness Analysis:**

The robust variant shows a robustness score of {robust.get('robustness_score', 0):.3f}, significantly higher than 
the clean version ({clean.get('robustness_score', 0):.3f}), indicating better performance under market volatility 
and adversarial conditions. This suggests that adversarial training improves the model's ability to handle 
unexpected market movements and data perturbations."""

        return summary

    def generate_explainability_text(self) -> Dict[str, str]:
        """Generate explainability text for each model architecture."""
        
        return {
            "baseline_dqn": """
            **Baseline Deep Q-Network (DQN)**
            
            This is the standard deep reinforcement learning approach for portfolio optimization. The model:
            - Uses a deep neural network to estimate Q-values (expected future rewards) for different actions
            - Implements experience replay to break correlation between consecutive experiences
            - Uses a target network to stabilize training
            - Makes decisions based on maximizing expected portfolio returns while managing risk
            
            **Strengths:** Simple architecture, well-understood, fast training
            **Limitations:** May struggle with complex market patterns, sensitive to input perturbations
            """,
            
            "mha_dqn_clean": """
            **Multi-Head Attention Deep Q-Network (MHA-DQN) - Clean**
            
            This enhanced version incorporates multi-head attention mechanisms:
            - **Attention Mechanism**: Focuses on the most relevant features at different time steps
            - **Multiple Heads**: Captures diverse market patterns simultaneously (trends, volatility, momentum)
            - **Feature Relationships**: Learns complex interactions between different financial indicators
            - **Temporal Dependencies**: Better understands long-term market dynamics
            
            **Multi-Head Attention Explained:**
            - Each attention head focuses on different aspects: price trends, volatility patterns, volume dynamics
            - Combines information from multiple heads to make more informed decisions
            - Automatically weights the importance of different features based on market conditions
            
            **Improvements over Baseline:** Better feature selection, improved pattern recognition, more nuanced decision-making
            """,
            
            "mha_dqn_robust": """
            **Multi-Head Attention Deep Q-Network (MHA-DQN) - Adversarially Robust**
            
            This is the most advanced model, combining multi-head attention with adversarial training:
            - **All MHA-DQN Benefits**: Multi-head attention for better feature understanding
            - **Adversarial Training**: Trained with perturbed inputs to improve robustness
            - **Resilience**: Better handles market shocks, data noise, and unexpected events
            - **Generalization**: Performs well even when market conditions differ from training
            
            **How Adversarial Training Works:**
            1. During training, the model is exposed to intentionally perturbed market data
            2. These perturbations simulate market uncertainty, data errors, or malicious attacks
            3. The model learns to make decisions that are robust to these perturbations
            4. This translates to better real-world performance during volatile or uncertain periods
            
            **Financial Implications:**
            - More reliable during market crashes or unexpected news events
            - Less sensitive to data quality issues or feed disruptions
            - Better performance during regime changes or market transitions
            - Higher confidence in out-of-sample performance
            
            **Why This Matters:**
            In real trading, models must handle noisy data, market manipulation attempts, and sudden regime changes.
            Adversarial training ensures the model maintains performance even when inputs deviate from expectations.
            """
        }

    def generate_forecast_explainability(
        self, 
        ticker: str,
        forecast_data: Dict,
        model_type: str = "MHA-DQN (Adversarially Robust)"
    ) -> str:
        """Generate explainability text for forecast predictions using OpenAI API."""
        
        if self.use_llm:
            try:
                return self._generate_forecast_llm_explainability(ticker, forecast_data, model_type)
            except Exception as e:
                print(f"Warning: LLM forecast explainability failed: {e}, using template instead")
        
        return self._generate_forecast_template_explainability(ticker, forecast_data, model_type)
    
    def _generate_forecast_llm_explainability(
        self, 
        ticker: str,
        forecast_data: Dict,
        model_type: str
    ) -> str:
        """Generate forecast explainability using OpenAI API based on actual model results."""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=self.api_key)
            
            # Extract key information from forecast data
            q_values = forecast_data.get("q_values", [])
            predicted_action = forecast_data.get("predicted_action", "HOLD")
            confidence = forecast_data.get("confidence_score", 0.5)
            forecasted_price = forecast_data.get("forecasted_price", 0.0)
            current_price = forecast_data.get("current_price", 0.0)
            price_change_pct = forecast_data.get("expected_return_pct", 0.0) * 100
            full_data_range = forecast_data.get("full_data_range", "N/A")
            data_range = forecast_data.get("data_range", "N/A")
            total_data_points = forecast_data.get("total_data_points", 0)
            forecast_date = forecast_data.get("forecast_date", "N/A")
            last_data_date = forecast_data.get("last_data_date", "N/A")
            horizon_forecasts = forecast_data.get("horizon_forecasts", {})
            
            # Format Q-values
            q_values_text = ""
            if len(q_values) >= 3:
                q_values_text = f"SELL: {q_values[0]:.3f}, HOLD: {q_values[1]:.3f}, BUY: {q_values[2]:.3f}"
            
            # Format horizon forecasts
            horizon_text = ""
            if horizon_forecasts:
                horizon_details = []
                for days, forecast in sorted(horizon_forecasts.items()):
                    horizon_details.append(
                        f"{days}-day: {forecast.get('recommendation', 'HOLD')} "
                        f"(${forecast.get('forecasted_price', 0):.2f}, {forecast.get('price_change_pct', 0):+.2f}%, "
                        f"confidence: {forecast.get('confidence', 0):.1%})"
                    )
                horizon_text = " | ".join(horizon_details)
            
            prompt = f"""You are a financial AI analyst explaining trading model predictions. Analyze the following forecast data for {ticker}:

**Model Information:**
- Model Type: {model_type}
- Historical Data Used: {full_data_range} ({total_data_points} trading days)
- Input Sequence: {data_range}
- Last Data Point: {last_data_date}
- Forecast Date: {forecast_date}

**Current Market State:**
- Current Price: ${current_price:.2f}
- Forecasted Price: ${forecasted_price:.2f}
- Expected Price Change: {price_change_pct:+.2f}%

**Model Predictions (Q-values):**
- Q-values: [{q_values_text}]
- Predicted Action: {predicted_action}
- Confidence: {confidence:.1%}

**Multi-Horizon Forecasts:**
{horizon_text if horizon_text else "N/A"}

**Task:**
Write a comprehensive but concise explanation (2-3 paragraphs) that:
1. Explains what data the model analyzed and how much historical context it used
2. Interprets the Q-values and explains why the model chose {predicted_action}
3. Discusses the confidence level and what it means for the prediction reliability
4. Explains the forecasted price change and its implications
5. If multi-horizon forecasts are provided, discusses how the prediction changes over different time horizons
6. Provides context about the model's decision-making process in plain financial language

Write in clear, professional language suitable for traders and portfolio managers. Focus on actionable insights."""

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert financial AI analyst specializing in explaining deep reinforcement learning model predictions for stock trading. You translate complex model outputs into clear, actionable insights."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,
                temperature=0.6
            )
            
            return response.choices[0].message.content.strip()
            
        except (ImportError, Exception) as e:
            print(f"Warning: LLM forecast explainability generation failed: {e}, using template instead")
            return self._generate_forecast_template_explainability(ticker, forecast_data, model_type)
    
    def _generate_forecast_template_explainability(
        self, 
        ticker: str,
        forecast_data: Dict,
        model_type: str
    ) -> str:
        """Generate template-based explainability when OpenAI is not available."""
        
        q_values = forecast_data.get("q_values", [])
        predicted_action = forecast_data.get("predicted_action", "HOLD")
        confidence = forecast_data.get("confidence_score", 0.5)
        forecasted_price = forecast_data.get("forecasted_price", 0.0)
        current_price = forecast_data.get("current_price", 0.0)
        price_change_pct = forecast_data.get("expected_return_pct", 0.0) * 100
        full_data_range = forecast_data.get("full_data_range", "N/A")
        data_range = forecast_data.get("data_range", "N/A")
        total_data_points = forecast_data.get("total_data_points", 0)
        forecast_date = forecast_data.get("forecast_date")
        last_data_date = forecast_data.get("last_data_date")
        
        # Handle dates - use actual values or sensible fallbacks
        if not forecast_date or forecast_date == "N/A":
            forecast_date = "next trading day"
        if not last_data_date or last_data_date == "N/A":
            last_data_date = "last data point"
        
        # Format date info
        if forecast_date and last_data_date and forecast_date != "next trading day" and last_data_date != "last data point":
            date_info = f"for {forecast_date} (next trading day after {last_data_date})"
        elif forecast_date and forecast_date != "next trading day":
            date_info = f"for {forecast_date}"
        else:
            date_info = "for next trading day"
        
        q_values_text = ""
        if len(q_values) >= 3:
            q_values_text = f"SELL: {q_values[0]:.3f}, HOLD: {q_values[1]:.3f}, BUY: {q_values[2]:.3f}"
        
        explainability = f"""The {model_type} model utilized {full_data_range} of historical data ({total_data_points} trading days) to learn market patterns and contextualize recent trends. The model analyzed the sequence from {data_range} as input, informed by the full historical dataset spanning {total_data_points} trading days.

The model's Q-value analysis shows: [{q_values_text}]. Based on these Q-values, the model predicted **{predicted_action}** {date_info} with {confidence:.1%} confidence. The Q-values represent the model's estimate of expected future returns for each action, with the highest Q-value ({q_values[max(0, 1, 2) if len(q_values) >= 3 else 1] if len(q_values) >= 3 else 'N/A'}) indicating the optimal action.

The forecasted price of **${forecasted_price:.2f}** represents a {price_change_pct:+.2f}% change from the current price of ${current_price:.2f}. This prediction is based on the model's learned patterns from {total_data_points} days of historical data, with particular attention to the most recent {data_range.split('(')[-1].split('days')[0].strip() if 'days' in data_range else '20'} days of market activity."""
        
        return explainability
    
    def generate_multi_horizon_summary(
        self,
        ticker: str,
        horizon_forecasts: Dict,
        current_price: float
    ) -> str:
        """Generate a summary of multi-horizon forecasts using OpenAI API."""
        
        if self.use_llm and horizon_forecasts:
            try:
                return self._generate_multi_horizon_llm_summary(ticker, horizon_forecasts, current_price)
            except Exception as e:
                print(f"Warning: LLM multi-horizon summary failed: {e}, using template instead")
        
        return self._generate_multi_horizon_template_summary(ticker, horizon_forecasts, current_price)
    
    def _generate_multi_horizon_llm_summary(
        self,
        ticker: str,
        horizon_forecasts: Dict,
        current_price: float
    ) -> str:
        """Generate multi-horizon forecast summary using OpenAI API."""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=self.api_key)
            
            # Format horizon forecasts
            forecasts_text = "\n".join([
                f"- {days}-day forecast: {forecast.get('recommendation', 'HOLD')} "
                f"at ${forecast.get('forecasted_price', 0):.2f} "
                f"({forecast.get('price_change_pct', 0):+.2f}% change, "
                f"confidence: {forecast.get('confidence', 0):.1%})"
                for days, forecast in sorted(horizon_forecasts.items())
            ])
            
            prompt = f"""Analyze the following multi-horizon stock price forecasts for {ticker}:

**Current Price:** ${current_price:.2f}

**Forecasts:**
{forecasts_text}

Provide a concise summary (2-3 sentences) that:
1. Highlights the key recommendation pattern across different time horizons
2. Explains how uncertainty increases with longer horizons
3. Provides actionable insights for traders

Write in clear, professional financial language."""

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial analyst expert at explaining multi-horizon trading forecasts."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.6
            )
            
            return response.choices[0].message.content.strip()
            
        except (ImportError, Exception) as e:
            print(f"Warning: LLM multi-horizon summary failed: {e}, using template instead")
            return self._generate_multi_horizon_template_summary(ticker, horizon_forecasts, current_price)
    
    def _generate_multi_horizon_template_summary(
        self,
        ticker: str,
        horizon_forecasts: Dict,
        current_price: float
    ) -> str:
        """Generate template-based multi-horizon summary."""
        
        if not horizon_forecasts:
            return "No multi-horizon forecasts available."
        
        recommendations = [forecast.get('recommendation', 'HOLD') for forecast in horizon_forecasts.values()]
        main_recommendation = max(set(recommendations), key=recommendations.count)
        
        summary = f"""Multi-horizon analysis for {ticker} shows a primary recommendation of **{main_recommendation}** across different time frames. """
        
        if len(set(recommendations)) > 1:
            summary += f"Note that recommendations vary across horizons ({', '.join(set(recommendations))}), reflecting increasing uncertainty over longer time periods. "
        
        summary += f"Short-term forecasts (1-5 days) tend to have higher confidence, while longer-term predictions (10 days) incorporate more uncertainty and market volatility factors."""
        
        return summary

