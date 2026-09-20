import unittest
from audit_lineup_decisions import sample_band, select_action, pd, APP, DATA, read


class DecisionTests(unittest.TestCase):
    def test_sample_is_not_a_form_label(self):
        self.assertEqual(sample_band([]),'sin calientes')
        self.assertEqual(sample_band([20,100]),'calientes con muestra pequeña')
        self.assertEqual(sample_band([70,100]),'calientes con muestra amplia')
        self.assertEqual(sample_band([100]),'muestra intermedia')

    def test_small_samples_fall_back(self):
        self.assertEqual(select_action(pd.DataFrame({'win':[0]*39,'baseP':[.7]*39})),'base')

    def test_no_duplicate_gain_when_base_already_picks_lower(self):
        self.assertEqual(select_action(pd.DataFrame({'win':[0]*50,'baseP':[.3]*50})),'base')
        self.assertEqual(select_action(pd.DataFrame({'win':[0]*50,'baseP':[.7]*50})),'inferior')

    def test_export_reproduces_all_comparisons(self):
        report=read(DATA/'lineup-decision-audit.json')
        rows=pd.read_csv(APP/'outputs/lineup-decisions/predictions.csv')
        self.assertTrue((rows.date>='2026-07-01').all())
        self.assertTrue((rows.date<report['asOf']).all())
        self.assertFalse(rows.gamePk.duplicated().any())
        for name,result in report['policies'].items():
            self.assertEqual(int((rows[name]==rows.win).sum()),result['after']['correct'])
            self.assertEqual(len(rows),result['games'])
        self.assertFalse(report['activated'])


if __name__=='__main__':unittest.main()
