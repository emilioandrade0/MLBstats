import unittest
import numpy as np
from low_confidence_specialist import apply_policy, predict_monthly, score, read, ROOT


class SpecialistTests(unittest.TestCase):
    def test_future_outcomes_and_features_cannot_change_earlier_predictions(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(640, 11))
        base = np.full(640, .52)
        dates = np.array(['2024-12-01']*600 + ['2025-01-02']*20 + ['2025-02-02']*20)
        y = np.arange(640) % 2
        candidate = ('test', 'test', 'niche', .1)
        before, folds = predict_monthly(x, base, y, dates, candidate)
        changed_y, changed_x = y.copy(), x.copy()
        changed_y[-20:] = 1-changed_y[-20:]
        changed_x[-20:] += 100
        after, _ = predict_monthly(changed_x, base, changed_y, dates, candidate)
        np.testing.assert_array_equal(before[600:620], after[600:620])
        self.assertTrue(all(f['trainedThrough'] < f['month']+'-01' for f in folds))

    def test_policy_keeps_unscored_games_and_reverses_both_sides(self):
        base = np.array([.52, .48, .8, .53])
        after = apply_policy(base, np.array([.4, .4, np.nan, .45]))
        np.testing.assert_allclose(after, [.4, .6, .8, .53])

    def test_paired_counts_include_damage_not_only_rescues(self):
        result = score(np.array([.52,.52,.48]), np.array([.4,.4,.48]),
            np.array([0,1,0]), np.array(['2026-01-01']*3), np.array([True]*3), bootstrap=True)
        self.assertEqual(result['rescued'], 1)
        self.assertEqual(result['damaged'], 1)
        self.assertEqual(result['netCorrect'], 0)
        self.assertEqual(result['games'], 3)

    def test_published_report_reconciles_and_is_never_active(self):
        report = read(ROOT/'public/data/low-confidence-specialist.json')
        measured = report['evaluation']
        self.assertFalse(report['selected']['activated'])
        self.assertEqual(report['coverage']['outsideNicheChanged'], 0)
        self.assertEqual(report['coverage']['keptFraction'], 1)
        self.assertEqual(measured['netCorrect'], measured['rescued']-measured['damaged'])
        self.assertEqual(measured['netCorrect'], measured['afterCorrect']-measured['beforeCorrect'])
        self.assertEqual(sum(m['games'] for m in report['monthly']), measured['games'])
        self.assertEqual(sum(m['netCorrect'] for m in report['monthly']), measured['netCorrect'])
        self.assertEqual(len(report['changes']), measured['changed'])
        chosen = min(report['candidates'], key=lambda r: (-r['design']['netCorrect'], r['design']['changed'], r['id']))
        self.assertEqual(chosen['id'], report['selected']['id'])


if __name__ == '__main__':
    unittest.main()
