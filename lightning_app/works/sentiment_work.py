from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import requests
from . import LightningWork, HAS_LIGHTNING

from ..config import ALPHA_VANTAGE_API_KEY, ALPHA_VANTAGE_URL, OPENAI_API_KEY

ALPHA_VANTAGE_NEWS_URL = ALPHA_VANTAGE_URL


class SentimentWork(LightningWork):
    """Fetches recent news sentiment for the requested ticker using Alpha Vantage and OpenAI."""

    def _get_openai_sentiment_and_summary(self, articles: List[Dict], alpha_vantage_score: float) -> tuple[Optional[float], Optional[str]]:
        """Get sentiment score and summary from OpenAI API based on recent news articles."""
        if not OPENAI_API_KEY or not articles:
            return None, None
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            
            # Get most recent articles (already sorted by latest, so first items are most recent)
            recent_articles = articles[:10] if len(articles) >= 10 else articles
            
            # Combine article summaries for analysis
            combined_text = "\n\n".join([
                f"Title: {art.get('title', '')}\nSummary: {art.get('summary', '')}\nPublished: {art.get('time_published', 'Unknown')}"
                for art in recent_articles
            ])
            
            if not combined_text:
                return None, None
            
            # Create enhanced prompt for sentiment analysis and summary
            # OpenAI will provide its own independent sentiment score
            prompt = f"""You are an expert financial sentiment analyst. Analyze the sentiment of the most recent news articles below and provide:

1. YOUR OWN INDEPENDENT sentiment score between -1 (very bearish) and +1 (very bullish), where 0 is neutral.
   - Base this score solely on your analysis of the news content, not on the Alpha Vantage score.
   - Consider factors like: positive/negative language, market impact, analyst opinions, company performance indicators, industry trends.
   - Be precise: use decimals (e.g., 0.65 for moderately bullish, -0.42 for moderately bearish).
   
2. A concise summary paragraph (2-3 sentences) that synthesizes:
   - Your sentiment analysis (your score)
   - The Alpha Vantage sentiment score ({alpha_vantage_score:.3f}) for comparison
   - Key themes or concerns from the recent news
   - Overall market sentiment implications

The summary should:
- Be written in clear, professional language for financial professionals
- Not list individual articles but provide a cohesive narrative
- Highlight any significant discrepancies or agreements between sentiment sources

Recent News Articles:
{combined_text}

Provide your response in this exact format:
SCORE: [your independent sentiment score between -1 and 1, as a decimal]
SUMMARY: [your summary paragraph here]"""
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert financial analyst specializing in market sentiment analysis. You synthesize information from multiple sources to provide actionable insights."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.4,
                max_tokens=300
            )
            
            # Parse the response
            content = response.choices[0].message.content.strip()
            
            # Extract score and summary
            score = None
            summary = None
            
            lines = content.split('\n')
            for line in lines:
                if line.startswith('SCORE:'):
                    try:
                        import re
                        numbers = re.findall(r'-?\d+\.?\d*', line)
                        if numbers:
                            score = float(numbers[0])
                            score = max(-1.0, min(1.0, score))  # Normalize to [-1, 1]
                    except (ValueError, IndexError):
                        pass
                elif line.startswith('SUMMARY:'):
                    summary = line.replace('SUMMARY:', '').strip()
                    # If summary spans multiple lines, get the rest
                    summary_idx = lines.index(line)
                    if summary_idx < len(lines) - 1:
                        remaining_lines = [l.strip() for l in lines[summary_idx + 1:] if l.strip() and not l.startswith('SCORE:')]
                        if remaining_lines:
                            summary += ' ' + ' '.join(remaining_lines)
            
            # Fallback: if parsing failed, try to extract from entire response
            if score is None:
                try:
                    import re
                    # Look for score in the response
                    numbers = re.findall(r'-?\d+\.?\d*', content)
                    for num in numbers:
                        potential_score = float(num)
                        if -1 <= potential_score <= 1:
                            score = potential_score
                            break
                except:
                    pass
            
            if summary is None:
                # Use entire content as summary if format not followed
                summary = content
            
            return score, summary
                
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower() or "insufficient_quota" in error_str.lower():
                print("\n⚠️  OpenAI API Quota Issue")
                print("   Your OpenAI API key has exceeded its quota limit.")
                print("   Check: https://platform.openai.com/usage")
                print("   Billing: https://platform.openai.com/account/billing")
                print("   Falling back to Alpha Vantage sentiment only.\n")
            elif "401" in error_str or "authentication" in error_str.lower() or "invalid" in error_str.lower():
                print("\n⚠️  OpenAI API Key Invalid")
                print("   → Get a new key: https://platform.openai.com/api-keys")
                print("   → Update: lightning_app/config.py\n")
            else:
                print(f"[WARNING] OpenAI sentiment analysis error: {e}")
            return None, None

    def run(self, ticker: str, items: int = 10) -> Dict[str, object]:
        api_key = ALPHA_VANTAGE_API_KEY

        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker.upper(),
            "sort": "LATEST",  # Get most recent news first
            "limit": 50,  # Get more to ensure we have recent news
            "apikey": api_key,
        }

        # Retry logic with exponential backoff for timeout issues
        import time
        max_retries = 3
        timeout_seconds = 60  # Increased timeout to 60 seconds
        
        for attempt in range(max_retries):
            try:
                response = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=timeout_seconds)
                response.raise_for_status()
                break  # Success, exit retry loop
            except requests.exceptions.Timeout as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 5  # Exponential backoff: 5s, 10s, 20s
                    print(f"[WARNING] Alpha Vantage news sentiment timeout (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"[ERROR] Alpha Vantage news sentiment timeout after {max_retries} attempts")
                    raise
            except requests.exceptions.RequestException as e:
                print(f"[ERROR] Alpha Vantage news sentiment request error: {e}")
                raise
        payload = response.json()
        feed: List[Dict[str, object]] = payload.get("feed", [])

        if not feed:
            return {
                "alpha_vantage_score": 0.0,
                "openai_score": None,
                "combined_score": 0.0,
                "summary": None
            }

        df = pd.DataFrame(feed)
        
        # Get only most recent news articles
        df["time_published"] = pd.to_datetime(df["time_published"], errors="coerce")
        df = df.sort_values("time_published", ascending=False)  # Most recent first
        recent_articles_df = df.head(items).copy()  # Get most recent N articles (copy to avoid SettingWithCopyWarning)
        
        # Calculate Alpha Vantage sentiment score from most recent news only
        recent_articles_df.loc[:, "overall_sentiment_score"] = pd.to_numeric(
            recent_articles_df["overall_sentiment_score"], errors="coerce"
        )
        alpha_vantage_score = recent_articles_df["overall_sentiment_score"].dropna().mean()
        
        if pd.isna(alpha_vantage_score):
            alpha_vantage_score = 0.0

        # Prepare articles for OpenAI analysis
        articles = recent_articles_df[
            ["title", "time_published", "summary", "overall_sentiment_label", "overall_sentiment_score"]
        ].to_dict("records")

        # Get OpenAI sentiment and summary
        openai_score, summary = self._get_openai_sentiment_and_summary(articles, alpha_vantage_score)
        
        # Calculate combined score (weighted average)
        # OpenAI provides independent sentiment analysis, so we weight them equally
        if openai_score is not None:
            # Equal weighting: both sources are valuable
            # Alpha Vantage: quantitative analysis
            # OpenAI: qualitative analysis with context understanding
            combined_score = (alpha_vantage_score * 0.5 + openai_score * 0.5)
            print(f"   → Combined sentiment: Alpha Vantage ({alpha_vantage_score:.3f}) + OpenAI ({openai_score:.3f}) = {combined_score:.3f}")
        else:
            combined_score = alpha_vantage_score
            print(f"   → Using Alpha Vantage sentiment only: {combined_score:.3f}")
        
        # If OpenAI summary is not available, create a simple template summary
        if summary is None:
            if alpha_vantage_score > 0.2:
                summary = f"Recent news sentiment for {ticker} is generally bullish (Alpha Vantage score: {alpha_vantage_score:.3f}), indicating positive market sentiment. Key themes from recent coverage suggest favorable developments and investor optimism."
            elif alpha_vantage_score < -0.2:
                summary = f"Recent news sentiment for {ticker} is generally bearish (Alpha Vantage score: {alpha_vantage_score:.3f}), indicating negative market sentiment. Key themes from recent coverage suggest challenges or concerns that may impact investor confidence."
            else:
                summary = f"Recent news sentiment for {ticker} is relatively neutral (Alpha Vantage score: {alpha_vantage_score:.3f}), with mixed signals from recent coverage. Market sentiment appears balanced between positive and negative factors."

        return {
            "alpha_vantage_score": float(alpha_vantage_score),
            "openai_score": float(openai_score) if openai_score is not None else None,
            "combined_score": float(combined_score),
            "summary": summary
        }
