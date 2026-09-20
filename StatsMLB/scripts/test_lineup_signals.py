import unittest
from audit_lineup_signals import summarize, bh, score_rule, fit_predict, pd, np, DATA, APP, read


class LineupTests(unittest.TestCase):
    def test_unknown_is_not_normal_and_order_is_respected(self):
        players=pd.DataFrame({'player_id':[1,2,3], 'batting_order':[100,500,300]})
        lookup={'1':{'kind':'batter','label':'racha caliente','talentTier':'elite','pa':60,'ops':.9},
                '2':{'kind':'batter','label':'normal','pa':20},'9':{'kind':'pitcher','label':'en problemas'}}
        f=summarize(players,lookup,9)
        self.assertEqual(f['hot'],1);self.assertEqual(f['normal'],1);self.assertEqual(f['unknown'],1)
        self.assertEqual(f['top4_normal'],0);self.assertEqual(f['top4_unknown'],1)
        self.assertEqual(f['elite_hot'],1);self.assertEqual(f['low_sample'],2)
        self.assertEqual(f['sp_trouble'],1)

    def test_multiple_testing_adjustment(self):
        np.testing.assert_allclose(bh([.01,.04,.9]),[.03,.06,.9])

    def test_validation_model_does_not_consume_test_labels(self):
        x=np.array([[0.],[1.],[2.],[3.]])
        p=fit_predict(x,np.array([0,0,1,1]),np.array([[1.5],[2.5]]))
        self.assertTrue(np.isfinite(p).all());self.assertTrue((p>0).all());self.assertTrue((p<1).all())
        self.assertGreater(p[1],p[0])

    def test_generated_cohort_and_pairs(self):
        report=read(DATA/'lineup-audit.json')
        frame=pd.read_csv(APP/'outputs/lineup-audit/lineup-games.csv')
        self.assertTrue((frame.snapshotCutoff<frame.date).all())
        self.assertTrue((frame.date<report['asOf']).all())
        self.assertTrue((frame.groupby('gamePk').size()==2).all())
        self.assertEqual(report['games'],report['discoveryGames']+report['validationGames'])
        self.assertFalse(report['activated'])
        self.assertEqual(report['rulesTested'],len(report['rules'])+len(report['lowSampleRules']))
        counts=[r['games'] for r in report['models'].values()]
        self.assertEqual(len(set(counts)),1)
        for r in report['rules']:
            self.assertGreaterEqual(r['discovery']['n'],100)
            self.assertLessEqual(r['validation'].get('marketResolved',0),r['validation']['n'])
        old=report['versusOldAlignmentSameGames']['lineup']
        self.assertEqual(old['before']['games'],old['after']['games'])

    def test_ambiguous_rules_abstain(self):
        frame=pd.DataFrame({'gamePk':[1,1]})
        self.assertEqual(score_rule(frame,np.array([True,True])),{'n':0})


if __name__=='__main__':unittest.main()
