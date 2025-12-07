from __future__ import annotations

from typing import Dict, Optional
import requests
from . import LightningWork, HAS_LIGHTNING
from ..config import ALPHA_VANTAGE_API_KEY, ALPHA_VANTAGE_URL, OPENAI_API_KEY, FMP_API_KEY


class FundamentalAnalysisWork(LightningWork):
    """Fetches and analyzes earnings calls, financial statements, and 10K filings using Alpha Vantage, FMP, and OpenAI API."""

    def __init__(self):
        if HAS_LIGHTNING:
            super().__init__(parallel=True)
        # If Lightning is not available, the base class handles initialization

    def _get_earnings_call_transcript(self, ticker: str) -> Optional[Dict]:
        """
        Get latest earnings call transcript and data.
        Tries Financial Modeling Prep (FMP) first for transcripts, falls back to Alpha Vantage for numbers.
        """
        # First, try to get earnings numbers from Alpha Vantage (always available)
        earnings_data = None
        try:
            print(f"[1] Fetching earnings data from Alpha Vantage for {ticker}...")
            params = {
                "function": "EARNINGS",
                "symbol": ticker.upper(),
                "apikey": ALPHA_VANTAGE_API_KEY,
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
                        print(f"[WARNING] Alpha Vantage earnings timeout (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"[ERROR] Alpha Vantage earnings timeout after {max_retries} attempts")
                        raise
                except requests.exceptions.RequestException as e:
                    print(f"[ERROR] Alpha Vantage earnings request error: {e}")
                    raise
            payload = response.json()
            
            # Alpha Vantage earnings data structure
            if "quarterlyEarnings" in payload:
                quarterly = payload["quarterlyEarnings"]
                if quarterly and len(quarterly) > 0:
                    # Get most recent earnings
                    latest = quarterly[0]
                    earnings_data = {
                        "date": latest.get("fiscalDateEnding", ""),
                        "reportedEPS": latest.get("reportedEPS", ""),
                        "estimatedEPS": latest.get("estimatedEPS", ""),
                        "surprise": latest.get("surprise", ""),
                        "surprisePercentage": latest.get("surprisePercentage", ""),
                        "transcript": None,  # Will be filled if we get transcript
                    }
                    print(f"[2] Earnings numbers fetched: {earnings_data.get('date')}")
        except Exception as e:
            print(f"[ERROR] Error fetching earnings data from Alpha Vantage: {e}")
        
        # Try to get transcript from Financial Modeling Prep (FMP) if API key is available
        transcript = None
        if FMP_API_KEY:
            transcript = self._get_earnings_transcript_fmp(ticker, earnings_data.get("date") if earnings_data else None)
            if transcript and earnings_data:
                earnings_data["transcript"] = transcript
                earnings_data["has_transcript"] = True
        
        # If no earnings data at all, return None
        if not earnings_data:
            return None
        
        return earnings_data
    
    def _get_earnings_transcript_fmp(self, ticker: str, earnings_date: Optional[str] = None) -> Optional[str]:
        """Get earnings call transcript from Financial Modeling Prep (FMP) API."""
        if not FMP_API_KEY:
            return None
        
        try:
            print(f"[3] Fetching earnings call transcript from FMP for {ticker}...")
            # FMP API endpoint for earnings transcripts - returns list of available transcripts
            url = "https://financialmodelingprep.com/api/v3/earning_call_transcript"
            params = {
                "symbol": ticker.upper(),
                "apikey": FMP_API_KEY,
            }
            
            # Retry logic for FMP API
            import time
            from datetime import datetime
            max_retries = 3
            timeout_seconds = 60
            
            for attempt in range(max_retries):
                try:
                    response = requests.get(url, params=params, timeout=timeout_seconds)
                    response.raise_for_status()
                    break  # Success, exit retry loop
                except requests.exceptions.Timeout as e:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 5
                        print(f"[WARNING] FMP earnings transcript timeout (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"[ERROR] FMP earnings transcript timeout after {max_retries} attempts")
                        raise
                except requests.exceptions.RequestException as e:
                    print(f"[ERROR] FMP earnings transcript request error: {e}")
                    raise
            
            transcripts_list = response.json()
            
            if not transcripts_list or len(transcripts_list) == 0:
                print(f"[WARNING] No transcripts found for {ticker} in FMP")
                return None
            
            # Try to match transcript with earnings date if available
            transcript_to_use = None
            if earnings_date:
                try:
                    earnings_dt = datetime.strptime(earnings_date, "%Y-%m-%d")
                    # Try to find transcript with matching date (within 30 days)
                    for transcript_info in transcripts_list:
                        transcript_date_str = transcript_info.get("date", "")
                        if transcript_date_str:
                            try:
                                transcript_dt = datetime.strptime(transcript_date_str[:10], "%Y-%m-%d")
                                days_diff = abs((transcript_dt - earnings_dt).days)
                                if days_diff <= 30:  # Match within 30 days
                                    transcript_to_use = transcript_info
                                    print(f"[INFO] Found matching transcript for earnings date {earnings_date}")
                                    break
                            except:
                                continue
                except:
                    pass
            
            # If no match or no date, use most recent transcript
            if not transcript_to_use:
                transcript_to_use = transcripts_list[0]
            
            # Extract year and quarter if available
            year = transcript_to_use.get("year")
            quarter = transcript_to_use.get("quarter")
            
            # Fetch the actual transcript content using year and quarter if available
            if year and quarter:
                url_detailed = f"https://financialmodelingprep.com/api/v3/earning_call_transcript/{ticker.upper()}"
                params_detailed = {
                    "year": year,
                    "quarter": quarter,
                    "apikey": FMP_API_KEY,
                }
                
                try:
                    response_detailed = requests.get(url_detailed, params=params_detailed, timeout=timeout_seconds)
                    response_detailed.raise_for_status()
                    transcript_data = response_detailed.json()
                    
                    if transcript_data and len(transcript_data) > 0:
                        transcript_text = transcript_data[0].get("content", "")
                    else:
                        # Fallback to content in list
                        transcript_text = transcript_to_use.get("content", "")
                except:
                    # Fallback to content in list
                    transcript_text = transcript_to_use.get("content", "")
            else:
                # Use content directly from list
                transcript_text = transcript_to_use.get("content", "")
            
            if transcript_text and len(transcript_text) > 100:
                print(f"[4] Earnings call transcript fetched: {len(transcript_text)} characters")
                return transcript_text
            else:
                print(f"[WARNING] Transcript too short or empty for {ticker}")
                return None
                
        except Exception as e:
            print(f"[ERROR] Error fetching transcript from FMP: {e}")
            return None

    def _get_earnings_call_score_openai(self, ticker: str, earnings_data: Dict) -> Optional[float]:
        """Generate a scaled sentiment score (-1 to +1) from earnings call transcript using OpenAI."""
        if not OPENAI_API_KEY:
            return None
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            
            # Build prompt for scoring - use transcript if available, otherwise just numbers
            has_transcript = earnings_data.get("has_transcript", False) and earnings_data.get("transcript")
            
            if has_transcript:
                transcript = earnings_data.get("transcript", "")
                # Truncate transcript to fit token limits (keep first 4000 chars for scoring)
                transcript_preview = transcript[:4000] + "..." if len(transcript) > 4000 else transcript
                
                prompt = f"""You are a financial sentiment analyst. Analyze the earnings call transcript below and provide ONLY a numerical sentiment score.

Company: {ticker}
Fiscal Date: {earnings_data.get('date', 'N/A')}
Reported EPS: {earnings_data.get('reportedEPS', 'N/A')}
Estimated EPS: {earnings_data.get('estimatedEPS', 'N/A')}
Surprise: {earnings_data.get('surprise', 'N/A')}
Surprise Percentage: {earnings_data.get('surprisePercentage', 'N/A')}%

Earnings Call Transcript:
{transcript_preview}

Provide a sentiment score from -1.0 (very bearish) to +1.0 (very bullish), where:
- -1.0 to -0.5: Very bearish (major concerns, negative outlook)
- -0.5 to -0.1: Moderately bearish (some concerns)
- -0.1 to +0.1: Neutral (mixed signals)
- +0.1 to +0.5: Moderately bullish (positive outlook)
- +0.5 to +1.0: Very bullish (strong positive signals)

Consider: management tone, forward guidance, revenue trends, profit margins, competitive position, market outlook, and overall confidence.

Output format: SCORE: [number between -1.0 and 1.0]
Example: SCORE: 0.65"""
            else:
                # If no transcript, score based on earnings numbers only
                surprise_pct = earnings_data.get("surprisePercentage", "0")
                try:
                    surprise_val = float(surprise_pct) if surprise_pct else 0.0
                except:
                    surprise_val = 0.0
                
                prompt = f"""You are a financial sentiment analyst. Analyze the earnings data below and provide ONLY a numerical sentiment score.

Company: {ticker}
Fiscal Date: {earnings_data.get('date', 'N/A')}
Reported EPS: {earnings_data.get('reportedEPS', 'N/A')}
Estimated EPS: {earnings_data.get('estimatedEPS', 'N/A')}
Surprise: {earnings_data.get('surprise', 'N/A')}
Surprise Percentage: {surprise_val}%

Note: Full earnings call transcript not available. Score based on earnings numbers only.

Provide a sentiment score from -1.0 (very bearish) to +1.0 (very bullish) based on:
- EPS surprise (positive surprise = bullish)
- Reported vs Estimated EPS comparison
- Overall earnings performance

Output format: SCORE: [number between -1.0 and 1.0]
Example: SCORE: 0.45"""
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial sentiment analyst. You provide ONLY numerical sentiment scores in the format 'SCORE: [number]'. Never provide additional text."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Lower temperature for more consistent scoring
                max_tokens=20  # Very short response - just the score
            )
            
            # Parse the score from response
            content = response.choices[0].message.content.strip()
            import re
            score_match = re.search(r'SCORE:\s*([-+]?\d*\.?\d+)', content, re.IGNORECASE)
            if score_match:
                score = float(score_match.group(1))
                # Clamp to [-1, 1] range
                score = max(-1.0, min(1.0, score))
                print(f"[INFO] Earnings call sentiment score generated: {score:.3f}")
                return score
            else:
                # Fallback: try to find any number in the response
                numbers = re.findall(r'[-+]?\d*\.?\d+', content)
                if numbers:
                    score = float(numbers[0])
                    score = max(-1.0, min(1.0, score))
                    print(f"[INFO] Earnings call sentiment score extracted: {score:.3f}")
                    return score
                else:
                    print(f"[WARNING] Could not parse score from OpenAI response: {content}")
                    return None
                    
        except Exception as e:
            print(f"[WARNING] OpenAI earnings call score error: {e}")
            return None
    
    def _analyze_earnings_with_openai(self, ticker: str, earnings_data: Dict) -> Optional[str]:
        """Analyze earnings call data using OpenAI API. Uses transcript if available, otherwise numbers only."""
        if not OPENAI_API_KEY:
            return None
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            
            # Build analysis prompt - use transcript if available, otherwise just numbers
            has_transcript = earnings_data.get("has_transcript", False) and earnings_data.get("transcript")
            
            if has_transcript:
                transcript = earnings_data.get("transcript", "")
                # Truncate transcript to fit token limits (keep first 2000 chars)
                transcript_preview = transcript[:2000] + "..." if len(transcript) > 2000 else transcript
                
                earnings_summary = f"""
                Company: {ticker}
                Fiscal Date: {earnings_data.get('date', 'N/A')}
                Reported EPS: {earnings_data.get('reportedEPS', 'N/A')}
                Estimated EPS: {earnings_data.get('estimatedEPS', 'N/A')}
                Surprise: {earnings_data.get('surprise', 'N/A')}
                Surprise Percentage: {earnings_data.get('surprisePercentage', 'N/A')}%
                
                Earnings Call Transcript (excerpt):
                {transcript_preview}
                """
            else:
                earnings_summary = f"""
                Company: {ticker}
                Fiscal Date: {earnings_data.get('date', 'N/A')}
                Reported EPS: {earnings_data.get('reportedEPS', 'N/A')}
                Estimated EPS: {earnings_data.get('estimatedEPS', 'N/A')}
                Surprise: {earnings_data.get('surprise', 'N/A')}
                Surprise Percentage: {earnings_data.get('surprisePercentage', 'N/A')}%
                
                Note: Full earnings call transcript not available. Analysis based on earnings numbers only.
                """
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial analyst specializing in earnings call analysis. Provide concise, actionable insights about earnings performance."
                    },
                    {
                        "role": "user",
                        "content": f"Analyze the following earnings data for {ticker} and provide a brief summary (2-3 sentences) highlighting key insights:\n\n{earnings_summary}"
                    }
                ],
                temperature=0.4,
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI earnings analysis error: {e}")
            return None

    def _get_10k_summary_with_openai(self, ticker: str) -> Optional[str]:
        """Generate 10K summary using OpenAI API based on company overview."""
        if not OPENAI_API_KEY:
            print(f"[WARNING] OpenAI API key not set - cannot generate 10K summary for {ticker}")
            return None
        
        try:
            # First get company overview from Alpha Vantage
            print(f"[5] Fetching company overview from Alpha Vantage for {ticker}...")
            params = {
                "function": "OVERVIEW",
                "symbol": ticker.upper(),
                "apikey": ALPHA_VANTAGE_API_KEY,
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
                        print(f"[WARNING] Alpha Vantage earnings timeout (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"[ERROR] Alpha Vantage earnings timeout after {max_retries} attempts")
                        raise
                except requests.exceptions.RequestException as e:
                    print(f"[ERROR] Alpha Vantage earnings request error: {e}")
                    raise
            overview = response.json()
            
            if "Error Message" in overview:
                error_msg = overview.get("Error Message", "Unknown error")
                print(f"[ERROR] Alpha Vantage API error for {ticker}: {error_msg}")
                return None
            if "Note" in overview:
                note = overview.get("Note", "API call limit reached")
                print(f"[WARNING] Alpha Vantage API note for {ticker}: {note}")
                return None
            
            if not overview or len(overview) == 0:
                print(f"[WARNING] Empty overview data received for {ticker}")
                return None
            
            print(f"[6] Company overview data received for {ticker}")
            
            # Extract key information
            company_info = {
                "name": overview.get("Name", ""),
                "sector": overview.get("Sector", ""),
                "industry": overview.get("Industry", ""),
                "description": overview.get("Description", ""),
                "market_cap": overview.get("MarketCapitalization", ""),
                "pe_ratio": overview.get("PERatio", ""),
                "dividend_yield": overview.get("DividendYield", ""),
                "52_week_high": overview.get("52WeekHigh", ""),
                "52_week_low": overview.get("52WeekLow", ""),
            }
            
            # Use OpenAI to generate a high-level 10K summary
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            
            # Round numbers to 2 decimal places
            try:
                pe_ratio_formatted = f"{float(company_info['pe_ratio']):.2f}" if company_info['pe_ratio'] and company_info['pe_ratio'] != "None" else "N/A"
            except:
                pe_ratio_formatted = str(company_info['pe_ratio'])
            
            try:
                if company_info['dividend_yield'] and company_info['dividend_yield'] != "None":
                    div_yield_val = float(company_info['dividend_yield'])
                    div_yield_formatted = f"{(div_yield_val * 100):.2f}%"
                else:
                    div_yield_formatted = "N/A"
            except:
                div_yield_formatted = str(company_info['dividend_yield'])
            
            try:
                year_low_formatted = f"${float(company_info['52_week_low']):.2f}" if company_info['52_week_low'] and company_info['52_week_low'] != "None" else "N/A"
            except:
                year_low_formatted = f"${company_info['52_week_low']}" if company_info['52_week_low'] != "None" else "N/A"
            
            try:
                year_high_formatted = f"${float(company_info['52_week_high']):.2f}" if company_info['52_week_high'] and company_info['52_week_high'] != "None" else "N/A"
            except:
                year_high_formatted = f"${company_info['52_week_high']}" if company_info['52_week_high'] != "None" else "N/A"
            
            prompt = f"""Based on the following company information for {ticker}, provide a structured 10K-style summary with three distinct sections:
            
            1. BACKGROUND: Describe the company's business model, core operations, sector, and industry position (2-3 sentences).
            2. OPPORTUNITIES: Identify key growth opportunities, competitive advantages, and market trends that benefit the company (2-3 sentences).
            3. RISKS: Highlight significant risks, challenges, and potential threats facing the company (2-3 sentences).

            Company: {company_info['name']}
            Sector: {company_info['sector']}
            Industry: {company_info['industry']}
            Description: {company_info['description'][:500]}...
            Market Cap: {company_info['market_cap']}
            P/E Ratio: {pe_ratio_formatted}
            Dividend Yield: {div_yield_formatted}
            52-Week Range: {year_low_formatted} - {year_high_formatted}
            
            Format the response with clear section headers: "BACKGROUND:", "OPPORTUNITIES:", and "RISKS:". Use professional 10K filing language."""
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial analyst expert at summarizing company information in the style of SEC 10K filings. Provide clear, professional summaries with BACKGROUND, OPPORTUNITIES, and RISKS sections."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            summary = response.choices[0].message.content.strip()
            print(f"[7] 10K summary generated successfully for {ticker}")
            return summary
        except Exception as e:
            error_str = str(e)
            print(f"[ERROR] OpenAI 10K summary error for {ticker}: {error_str}")
            
            # If OpenAI fails, generate a basic summary from Alpha Vantage data only
            if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
                print(f"[8] OpenAI quota exceeded, generating fallback summary from Alpha Vantage data")
                return self._generate_fallback_summary(overview, ticker)
            
            import traceback
            traceback.print_exc()
            # Try fallback even for other errors - need to re-fetch overview if not in scope
            try:
                params = {
                    "function": "OVERVIEW",
                    "symbol": ticker.upper(),
                    "apikey": ALPHA_VANTAGE_API_KEY,
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
                            print(f"[WARNING] Alpha Vantage company overview timeout (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            print(f"[ERROR] Alpha Vantage company overview timeout after {max_retries} attempts")
                            raise
                    except requests.exceptions.RequestException as e:
                        print(f"[ERROR] Alpha Vantage company overview request error: {e}")
                        raise
                if response.status_code == 200:
                    overview_fallback = response.json()
                    if overview_fallback and not ("Error Message" in overview_fallback or "Note" in overview_fallback):
                        return self._generate_fallback_summary(overview_fallback, ticker)
            except:
                pass
            return None
    
    def _generate_fallback_summary(self, overview: Dict, ticker: str) -> str:
        """Generate a basic 10K-style summary from Alpha Vantage overview data when OpenAI is unavailable."""
        try:
            name = overview.get("Name", ticker)
            sector = overview.get("Sector", "N/A")
            industry = overview.get("Industry", "N/A")
            description = overview.get("Description", "")
            market_cap = overview.get("MarketCapitalization", "N/A")
            pe_ratio = overview.get("PERatio", "N/A")
            dividend_yield = overview.get("DividendYield", "N/A")
            year_high = overview.get("52WeekHigh", "N/A")
            year_low = overview.get("52WeekLow", "N/A")
            
            # Format market cap if numeric
            try:
                if market_cap and market_cap != "N/A":
                    mc_val = float(market_cap)
                    if mc_val >= 1e12:
                        market_cap_str = f"${mc_val/1e12:.2f}T"
                    elif mc_val >= 1e9:
                        market_cap_str = f"${mc_val/1e9:.2f}B"
                    elif mc_val >= 1e6:
                        market_cap_str = f"${mc_val/1e6:.2f}M"
                    else:
                        market_cap_str = f"${mc_val:,.0f}"
                else:
                    market_cap_str = "N/A"
            except:
                market_cap_str = market_cap
            
            # Format dividend yield (2 decimal places)
            try:
                if dividend_yield and dividend_yield != "N/A":
                    div_yield_str = f"{float(dividend_yield)*100:.2f}%"
                else:
                    div_yield_str = "N/A"
            except:
                div_yield_str = dividend_yield
            
            # Format 52-week high/low (2 decimal places)
            try:
                if year_high and year_high != "N/A":
                    year_high_str = f"${float(year_high):.2f}"
                else:
                    year_high_str = "N/A"
            except:
                year_high_str = str(year_high) if year_high != "None" else "N/A"
            
            try:
                if year_low and year_low != "N/A":
                    year_low_str = f"${float(year_low):.2f}"
                else:
                    year_low_str = "N/A"
            except:
                year_low_str = str(year_low) if year_low != "None" else "N/A"
            
            # Create summary from description and key metrics
            if description and len(description) > 50:
                desc_summary = description[:300] + "..." if len(description) > 300 else description
            else:
                desc_summary = f"{name} operates in the {sector} sector within the {industry} industry."
            
            # Format P/E ratio (2 decimal places)
            try:
                if pe_ratio and pe_ratio != "N/A":
                    pe_ratio_str = f"{float(pe_ratio):.2f}"
                else:
                    pe_ratio_str = "N/A"
            except:
                pe_ratio_str = str(pe_ratio)
            
            summary = f"BACKGROUND: {desc_summary} "
            summary += f"The company has a market capitalization of {market_cap_str} and trades at a P/E ratio of {pe_ratio_str}. "
            summary += f"Dividend yield stands at {div_yield_str}. "
            summary += f"Over the past 52 weeks, the stock has traded between {year_low_str} and {year_high_str}. "
            summary += f"\n\nOPPORTUNITIES: The company operates in the {industry} industry within the {sector} sector, which presents growth opportunities. "
            summary += f"Key competitive advantages include market position and industry leadership. "
            summary += f"\n\nRISKS: Market volatility and industry-specific challenges pose risks. "
            summary += f"Investors should consider regulatory changes, competitive pressures, and macroeconomic factors."
            
            print(f"[9] Fallback 10K summary generated from Alpha Vantage data")
            return summary
        except Exception as e:
            print(f"[ERROR] Error generating fallback summary: {e}")
            return None

    def _get_financial_statements_fmp(self, ticker: str) -> Optional[Dict]:
        """Get financial statements from FMP (Income Statement, Balance Sheet, Cash Flow)."""
        if not FMP_API_KEY:
            return None
        
        try:
            print(f"[FMP-1] Fetching financial statements from FMP for {ticker}...")
            import time
            max_retries = 3
            timeout_seconds = 60
            
            financial_data = {}
            
            # Fetch Income Statement (most recent annual)
            try:
                url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker.upper()}"
                params = {"limit": 1, "apikey": FMP_API_KEY}
                
                for attempt in range(max_retries):
                    try:
                        response = requests.get(url, params=params, timeout=timeout_seconds)
                        response.raise_for_status()
                        income_data = response.json()
                        if income_data and len(income_data) > 0:
                            financial_data["income_statement"] = income_data[0]
                            print(f"[FMP-2] Income statement fetched for {ticker}")
                        break
                    except requests.exceptions.Timeout:
                        if attempt < max_retries - 1:
                            time.sleep((2 ** attempt) * 5)
                        else:
                            raise
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                print(f"[WARNING] Could not fetch income statement: {e}")
            
            # Fetch Balance Sheet
            try:
                url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker.upper()}"
                params = {"limit": 1, "apikey": FMP_API_KEY}
                
                for attempt in range(max_retries):
                    try:
                        response = requests.get(url, params=params, timeout=timeout_seconds)
                        response.raise_for_status()
                        balance_data = response.json()
                        if balance_data and len(balance_data) > 0:
                            financial_data["balance_sheet"] = balance_data[0]
                            print(f"[FMP-3] Balance sheet fetched for {ticker}")
                        break
                    except requests.exceptions.Timeout:
                        if attempt < max_retries - 1:
                            time.sleep((2 ** attempt) * 5)
                        else:
                            raise
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                print(f"[WARNING] Could not fetch balance sheet: {e}")
            
            # Fetch Cash Flow Statement
            try:
                url = f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker.upper()}"
                params = {"limit": 1, "apikey": FMP_API_KEY}
                
                for attempt in range(max_retries):
                    try:
                        response = requests.get(url, params=params, timeout=timeout_seconds)
                        response.raise_for_status()
                        cashflow_data = response.json()
                        if cashflow_data and len(cashflow_data) > 0:
                            financial_data["cash_flow"] = cashflow_data[0]
                            print(f"[FMP-4] Cash flow statement fetched for {ticker}")
                        break
                    except requests.exceptions.Timeout:
                        if attempt < max_retries - 1:
                            time.sleep((2 ** attempt) * 5)
                        else:
                            raise
            except Exception as e:
                print(f"[WARNING] Could not fetch cash flow statement: {e}")
            
            return financial_data if financial_data else None
            
        except Exception as e:
            print(f"[ERROR] Error fetching financial statements from FMP: {e}")
            return None
    
    def _get_earnings_calendar_fmp(self, ticker: str) -> Optional[Dict]:
        """Get upcoming earnings calendar from FMP."""
        if not FMP_API_KEY:
            return None
        
        try:
            print(f"[FMP-5] Fetching earnings calendar from FMP for {ticker}...")
            url = "https://financialmodelingprep.com/api/v3/earnings_calendar"
            params = {
                "symbol": ticker.upper(),
                "apikey": FMP_API_KEY,
            }
            
            import time
            max_retries = 3
            timeout_seconds = 60
            
            for attempt in range(max_retries):
                try:
                    response = requests.get(url, params=params, timeout=timeout_seconds)
                    response.raise_for_status()
                    calendar_data = response.json()
                    
                    if calendar_data and len(calendar_data) > 0:
                        # Get upcoming earnings (future dates)
                        from datetime import datetime
                        upcoming = []
                        for event in calendar_data:
                            date_str = event.get("date", "")
                            if date_str:
                                try:
                                    event_date = datetime.strptime(date_str, "%Y-%m-%d")
                                    if event_date >= datetime.now():
                                        upcoming.append(event)
                                except:
                                    continue
                        
                        if upcoming:
                            print(f"[FMP-6] Found {len(upcoming)} upcoming earnings events")
                            return {
                                "upcoming": upcoming[:3],  # Next 3 upcoming
                                "all": calendar_data[:5]  # Recent 5
                            }
                        else:
                            # No upcoming, return most recent
                            print(f"[FMP-6] No upcoming earnings, returning most recent")
                            return {"recent": calendar_data[:3]}
                    
                    break
                except requests.exceptions.Timeout:
                    if attempt < max_retries - 1:
                        time.sleep((2 ** attempt) * 5)
                    else:
                        raise
                except requests.exceptions.RequestException as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(1)
            
            return None
            
        except Exception as e:
            print(f"[WARNING] Could not fetch earnings calendar: {e}")
            return None

    def run(self, ticker: str) -> Dict:
        """Get earnings call analysis, 10K summary, and additional FMP data."""
        print(f"\n[START] Starting fundamental analysis for {ticker}...")
        
        earnings_data = self._get_earnings_call_transcript(ticker)
        earnings_analysis = None
        earnings_call_score = None
        if earnings_data:
            # Generate both analysis text and numerical score
            earnings_analysis = self._analyze_earnings_with_openai(ticker, earnings_data)
            earnings_call_score = self._get_earnings_call_score_openai(ticker, earnings_data)
            
            # Add the score to earnings_data for feature engineering
            if earnings_call_score is not None:
                earnings_data["earnings_call_score"] = earnings_call_score
                print(f"[INFO] Earnings call score added to earnings_data: {earnings_call_score:.3f}")
        
        tenk_summary = self._get_10k_summary_with_openai(ticker)
        
        # Fetch additional FMP data if API key is available
        fmp_financials = None
        fmp_earnings_calendar = None
        if FMP_API_KEY:
            fmp_financials = self._get_financial_statements_fmp(ticker)
            fmp_earnings_calendar = self._get_earnings_calendar_fmp(ticker)
        
        result = {
            "earnings_analysis": earnings_analysis,
            "earnings_data": earnings_data,
            "10k_summary": tenk_summary,
            "fmp_financials": fmp_financials,  # Income, Balance Sheet, Cash Flow
            "fmp_earnings_calendar": fmp_earnings_calendar,  # Upcoming/recent earnings dates
        }
        
        print(f"[COMPLETE] Fundamental analysis complete for {ticker}:")
        print(f"  [INFO] 10K Summary: {'Generated' if tenk_summary else 'Not available'}")
        print(f"  [INFO] Earnings Analysis: {'Generated' if earnings_analysis else 'Not available'}")
        print(f"  [INFO] Earnings Transcript: {'Yes' if (earnings_data and earnings_data.get('has_transcript')) else 'No'}")
        print(f"  [INFO] FMP Financial Statements: {'Yes' if fmp_financials else 'No'}")
        print(f"  [INFO] FMP Earnings Calendar: {'Yes' if fmp_earnings_calendar else 'No'}")
        
        return result

