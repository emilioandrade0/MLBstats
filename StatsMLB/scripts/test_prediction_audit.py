"""Run: python -m unittest discover -s scripts -p test_prediction_audit.py"""
import unittest
import numpy as np
from audit_prediction_engines import DATA, APP, read, metrics, paired, pd


class AuditTests(unittest.TestCase):
    def test_metrics_missing_and_exact(self):
        m = metrics(np.array([.8,.2,np.nan]),np.array([1,0,1]),np.array(['a','a','b']))
        self.assertEqual(m['games'],2)
        self.assertEqual(m['accuracy'],1)
        self.assertAlmostEqual(m['brier'],.04)

    def test_paired_same_games(self):
        p = paired(np.array([.8,.8,.2]),np.array([.8,.2,np.nan]),np.array([1,0,0]),np.array(['a','b','c']),np.ones(3,dtype=bool))
        self.assertEqual(p['games'],2)
        self.assertEqual(p['netCorrect'],1)
        self.assertEqual(p['before']['games'],p['after']['games'])

    def test_export_reproduces_measured_2026(self):
        report = read(DATA/'prediction-audit.json')
        rows = pd.read_csv(APP/'outputs/prediction-audit/before-after.csv')
        rows = rows[rows.eligible & (rows.date >= '2026-01-01')]
        self.assertTrue((rows.date < report['asOf']).all())
        self.assertEqual(len(rows.gamePk.unique()),len(rows))
        for column,key in [('before','before'),('after','after')]:
            correct = int(((rows[column]>=.5)==rows.homeWin).sum())
            self.assertEqual(correct,report['beforeAfter']['validation2026'][key]['correct'])
        self.assertFalse(report['selected']['activated'])
        self.assertEqual(report['coverage']['canonicalScoreMismatches'],0)
        self.assertEqual(report['coverage']['snapshotBoundaryViolations'],0)


if __name__ == '__main__':
    unittest.main()
