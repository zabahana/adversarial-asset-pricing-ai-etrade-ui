try:
    from lightning.app import LightningApp
except ImportError:
    from lightning import LightningApp

from .flows.orchestrator import OrchestratorFlow


def main() -> None:
    """Entrypoint for the Lightning application."""
    app = LightningApp(OrchestratorFlow())
    app.run()


if __name__ == "__main__":
    main()
