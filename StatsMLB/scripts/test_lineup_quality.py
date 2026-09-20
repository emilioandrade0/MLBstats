import unittest
from audit_lineup_quality import attach_prior_counts, ops, pd, np, summarize


class QualityTests(unittest.TestCase):
    def test_same_day_and_future_results_are_excluded(self):
        starters=pd.DataFrame({'player_id':[1,1,2],'date':['2026-06-02','2026-06-03','2026-06-02']})
        daily=pd.DataFrame({'player_id':[1,1,1,2],'date':['2026-06-01','2026-06-02','2026-06-04','2026-06-02'],'hits':[2.,100.,200.,20.]})
        matched=attach_prior_counts(starters,daily)
        one=matched[matched.player_id==1].sort_values('date')
        self.assertEqual(one.hits.tolist(),[2.,100.])
        self.assertTrue(matched[matched.player_id==2].hits.isna().all())

    def test_ops_counts(self):
        result=ops(np.array([[100.,30.,50.,10.,2.,3.,115.]]))
        self.assertAlmostEqual(result[0],42/115+.5)

    def test_lower_side_is_opposite(self):
        d=pd.DataFrame({'win':[1,0,0],'baseP':[.6,.7,.8],'date':['a','b','c'],'runsFor':[5,2,1],'runsAgainst':[3,4,3]})
        result=summarize(d)
        self.assertEqual(result['lowerWins'],2)
        self.assertAlmostEqual(result['expectedLower'],.3)


if __name__=='__main__':unittest.main()
