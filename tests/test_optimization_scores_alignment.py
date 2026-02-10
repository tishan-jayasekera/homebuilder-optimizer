import unittest
import pandas as pd
import numpy as np

from src.optimization_engine import ReferralOptimizationEngine, LagMetrics, PacingMetrics


class TestOptimizationScoresAlignment(unittest.TestCase):
    def setUp(self):
        self.events = pd.DataFrame({
            "LeadId": [1, 2, 3, 10, 11],
            "Original Deal ID": [1, 1, 1, 10, 11],
            "Deals: Id": [1, 2, 3, 10, 11],
            "MediaPayer_BuilderRegionKey": ["P1", "P1", "P1", "P2", "P2"],
            "Dest_BuilderRegionKey": ["P1", "P1", "P1", "P2", "P2"],
            "lead_date": pd.to_datetime([
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-01",
                "2024-01-02",
            ]),
            "RefDate": pd.to_datetime([
                "2024-01-05",
                "2024-01-06",
                "2024-01-07",
                pd.NaT,
                pd.NaT,
            ]),
            "is_origin": [True, False, False, True, True],
            "is_referral": [False, True, True, False, False],
            "MediaCost_referral_event": [40.0, 30.0, 30.0, 100.0, 100.0],
            "utm_campaign": ["C1", "C1", "C1", "C2", "C2"],
        })
        self.media_raw = pd.DataFrame({
            "Campaign: Campaign name": ["C1"],
            "Conversions: All On-Facebook Leads - Total": [10],
        })
        self.engine = ReferralOptimizationEngine(
            events_df=self.events,
            origin_perf_df=pd.DataFrame(),
            media_raw_df=self.media_raw,
            lite=True,
        )

    def test_creative_metrics_spend_rm_conversion(self):
        metrics = self.engine.compute_payer_metrics(metrics_mode="creative")
        metrics = metrics.set_index("payer")

        p1 = metrics.loc["P1"]
        p2 = metrics.loc["P2"]

        self.assertAlmostEqual(p1["spend"], 100.0)
        self.assertAlmostEqual(p2["spend"], 200.0)

        self.assertAlmostEqual(p1["rm"], 2.0)
        self.assertAlmostEqual(p2["rm"], 0.0)

        self.assertAlmostEqual(p1["conversion"], 0.1)
        self.assertTrue(np.isnan(p2["conversion"]))

    def test_total_score_renormalizes_when_conversion_missing(self):
        lag_metrics = LagMetrics(L_conv=0.0, L_ref=0.0, L_media=0)
        pacing_metrics = PacingMetrics(
            current_pacing_factor=1.0,
            status="Healthy",
            cumulative_actual=0,
            cumulative_target=0,
        )
        scores = self.engine.compute_optimization_scores(
            lag_metrics,
            pacing_metrics,
            metrics_mode="creative",
        )
        score_map = {s.payer: s for s in scores}

        p1_score = score_map["P1"].total_score
        p2_score = score_map["P2"].total_score

        self.assertAlmostEqual(p1_score, 66.5, places=1)
        self.assertAlmostEqual(p2_score, 35.29, places=2)


if __name__ == "__main__":
    unittest.main()
