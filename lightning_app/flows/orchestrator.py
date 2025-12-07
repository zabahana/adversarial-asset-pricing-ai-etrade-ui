try:
    from lightning.app import LightningFlow
except ImportError:
    from lightning import LightningFlow

from ..config import DEFAULT_TICKER
from ..works.data_fetch_work import DataFetchWork
from ..works.feature_engineering_work import FeatureEngineeringWork
from ..works.model_inference_work import ModelInferenceWork
from ..works.sentiment_work import SentimentWork
from ..works.macro_work import MacroWork
from ..ui.dashboards import PortfolioDashboard


class OrchestratorFlow(LightningFlow):
    """Coordinates data retrieval, feature prep, inference, and UI updates."""

    def __init__(self) -> None:
        super().__init__()
        self.data_work = DataFetchWork(cache_dir="results/cached_history")
        self.feature_work = FeatureEngineeringWork(cache_dir="results/cached_features")
        self.model_work = ModelInferenceWork(
            model_dir="models",
            results_dir="results",
        )
        self.sentiment_work = SentimentWork()
        self.macro_work = MacroWork()
        self.dashboard = PortfolioDashboard()
        self.ticker: str = DEFAULT_TICKER

    def run(self) -> None:
        """Execute the end-to-end workflow for the configured ticker."""
        price_path = self.data_work.run(self.ticker)
        features_path = self.feature_work.run(price_path, self.ticker)
        model_results_path = self.model_work.run(self.ticker, features_path)
        sentiment_payload = self.sentiment_work.run(self.ticker)
        macro_payload = self.macro_work.run()

        self.dashboard.set_data(
            price_path=price_path,
            model_results_path=model_results_path,
            sentiment_payload=sentiment_payload,
            macro_payload=macro_payload,
        )

    def configure_layout(self):
        return self.dashboard
