from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import plotly.express as px
try:
    from lightning.app import LightningComponent
except ImportError:
    from lightning import LightningComponent


class PortfolioDashboard(LightningComponent):
    """Streamlit-based dashboard for interacting with inference outputs."""

    def __init__(self) -> None:
        super().__init__()
        self._price_path: Optional[Path] = None
        self._model_results_path: Optional[Path] = None
        self._sentiment_payload: Optional[Dict[str, object]] = None
        self._macro_payload: Optional[Dict[str, object]] = None

    def set_data(
        self,
        price_path: str,
        model_results_path: str,
        sentiment_payload: Dict[str, object],
        macro_payload: Dict[str, object],
    ) -> None:
        self._price_path = Path(price_path)
        self._model_results_path = Path(model_results_path)
        self._sentiment_payload = sentiment_payload
        self._macro_payload = macro_payload

    def render(self):
        import streamlit as st

        st.title("Adversarial-Robust Asset Pricing Intelligence")
        st.caption("Compare baseline DQN against adversarially trained MHA-DQN models.")
        
        # Display ticker info (can be enhanced later to accept input)
        if hasattr(self, '_current_ticker'):
            st.sidebar.subheader(f"Current Ticker: {self._current_ticker}")
        
        st.info("💡 To change the ticker, modify `self.ticker` in the orchestrator or restart the app with a different ticker.")

        if self._price_path and self._price_path.exists():
            price_df = pd.read_parquet(self._price_path)
            price_df = price_df.rename(columns={"5. adjusted close": "close"})
            price_df["close"] = price_df["close"].astype(float)
            fig = px.line(price_df, x=price_df.index, y="close", title="5-Year Price History")
            st.plotly_chart(fig, use_container_width=True)

        if self._model_results_path and self._model_results_path.exists():
            payload = json.load(self._model_results_path.open())
            st.header("Model Comparison")
            for name, content in payload.items():
                st.subheader(name.replace("_", " ").title())
                st.table(pd.DataFrame(content["metrics"], index=[0]))

        if self._sentiment_payload:
            st.header("News Sentiment")
            st.metric("Rolling Sentiment Score", round(self._sentiment_payload["overall"], 3))
            for article in self._sentiment_payload.get("articles", [])[:5]:
                st.write(f"**{article['title']}** — {article['overall_sentiment_label']}")
                st.caption(article.get("summary", ""))

        if self._macro_payload:
            st.header("Macro Backdrop")
            for key, info in self._macro_payload.items():
                description = info.get("description", key.title())
                latest = info.get("latest")
                st.metric(description, latest)
