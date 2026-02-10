import unittest
import pandas as pd

from src.referral_logic import count_leads_refs, lead_ref_masks, prepare_referral_ids


class TestReferralLogicCounts(unittest.TestCase):
    def test_count_leads_refs_with_deal_ids(self):
        df = pd.DataFrame({
            "Original Deal ID": [1, 1, 1],
            "Deals: Id": [1, 2, 3],
            "LeadId": [1, 2, 3],
            "MediaPayer_BuilderRegionKey": ["A", "A", "A"],
            "Dest_BuilderRegionKey": ["A", "A", "A"],
        })
        df_prepped, _, _, _ = prepare_referral_ids(df, inplace=False)
        leads, refs, events, orig_set = count_leads_refs(df_prepped)
        self.assertEqual(leads, 1)
        self.assertEqual(refs, 2)
        self.assertEqual(events, 3)
        lead_mask, ref_mask = lead_ref_masks(df_prepped, orig_set)
        self.assertEqual(int(lead_mask.sum()), 1)
        self.assertEqual(int(ref_mask.sum()), 2)

    def test_count_leads_refs_fallback(self):
        df = pd.DataFrame({
            "LeadId": [1, 2, 3, 4],
            "is_referral": [False, True, False, False],
            "MediaPayer_BuilderRegionKey": ["A", "A", "A", "B"],
            "Dest_BuilderRegionKey": ["A", "A", "B", "B"],
        })
        leads, refs, events, orig_set = count_leads_refs(df)
        self.assertIsNone(orig_set)
        self.assertEqual(leads, 2)
        self.assertEqual(refs, 2)
        self.assertEqual(events, 4)
        lead_mask, ref_mask = lead_ref_masks(df, orig_set)
        self.assertEqual(int(lead_mask.sum()), 2)
        self.assertEqual(int(ref_mask.sum()), 2)


if __name__ == "__main__":
    unittest.main()
