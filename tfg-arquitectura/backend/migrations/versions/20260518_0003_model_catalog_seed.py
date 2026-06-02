"""seed model catalog: add arima/auto-arima/sarimax, deactivate not-a-model

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-18

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_NEW_MODELS = [
    {
        "slug": "arima",
        "name": "ARIMA(1,1,1)",
        "model_type": "arima",
        "description": (
            "Non-seasonal ARIMA(1,1,1) baseline. Matches the 'arima' condition "
            "from the rolling back-test (04_backtesting_rolling.py)."
        ),
        "supports_mcp": False,
        "is_active": True,
    },
    {
        "slug": "auto-arima",
        "name": "AutoARIMA (AIC stepwise)",
        "model_type": "arima",
        "description": (
            "pmdarima auto_arima: selects p,d,q by AIC stepwise search. "
            "Non-seasonal; use sarima for seasonal models."
        ),
        "supports_mcp": False,
        "is_active": True,
    },
    {
        "slug": "sarimax",
        "name": "SARIMAX + ECB rates",
        "model_type": "arima",
        "description": (
            "SARIMA(0,1,1)(0,1,1)12 with ECB Deposit Facility Rate (dfr) and "
            "Main Refinancing Rate (mrr) as exogenous regressors."
        ),
        "supports_mcp": False,
        "is_active": True,
    },
]


def upgrade() -> None:
    conn = op.get_bind()

    # Insert new models (skip if slug already exists)
    for m in _NEW_MODELS:
        conn.execute(
            sa.text(
                """
                INSERT INTO model_catalog (slug, name, model_type, description, supports_mcp, is_active)
                VALUES (:slug, :name, CAST(:model_type AS model_type), :description, :supports_mcp, :is_active)
                ON CONFLICT (slug) DO NOTHING
                """
            ),
            m,
        )

    # Deactivate test artifact
    conn.execute(
        sa.text("UPDATE model_catalog SET is_active = false WHERE slug = 'not-a-model'")
    )


def downgrade() -> None:
    conn = op.get_bind()
    for m in _NEW_MODELS:
        conn.execute(
            sa.text("DELETE FROM model_catalog WHERE slug = :slug"),
            {"slug": m["slug"]},
        )
    conn.execute(
        sa.text("UPDATE model_catalog SET is_active = true WHERE slug = 'not-a-model'")
    )
