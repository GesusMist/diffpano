"""Meaningful offline-study tests; all votes below are synthetic fixtures only."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from .common import write, read, digest
from . import human


class HumanStudyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.run=Path(self.temp.name)
        self.scenes=[dict(prompt_id='confirmation/test%02d'%i,scene_id='test%02d'%i) for i in range(24)]
        write(self.run/'real_dataset_manifest.json',dict(rows=self.scenes))

    def tearDown(self):self.temp.cleanup()

    def fixture(self,ready=True):
        study_id,public,private=human.assignments(self.scenes)
        for row in public:row['state']='ready' if ready else 'pending_assets_or_reviewed_caption'
        write(self.run/'human/assignments.json',dict(assignments=public))
        write(self.run/'human/private/method_key.json',dict(assignments=private))
        return public,private

    def csv(self,rows,name='fixture.csv'):
        path=self.run/name
        with path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=human.FIELDS);writer.writeheader();writer.writerows(rows)
        return path

    def vote(self,row,choice='tie'):
        return dict(assignment_id=row['assignment_id'],participant_id='fixture-rater-%d'%row['rater_slot'],rater_slot=str(row['rater_slot']),
            quality=choice,geometry_continuity=choice,prompt_adherence=choice,yaw='180',pitch='0',elapsed_seconds='12.5',
            submitted_at_utc='2026-10-26T00:00:00Z',protocol_sha256=row['protocol_sha256'])

    def test_assignment_balance_and_blinding(self):
        unused,public,private=human.assignments(self.scenes)
        self.assertEqual(len(public),240);self.assertEqual(len({row['assignment_id'] for row in public}),240)
        for slot in range(1,6):
            rows=[row for row in public if row['rater_slot']==slot]
            self.assertEqual(len(rows),48);self.assertEqual(sum(private[row['assignment_id']]['left_method']=='select' for row in rows),24)
        for scene in self.scenes:
            rows=[row for row in private.values() if row['prompt_id']==scene['prompt_id']]
            for baseline in ('sphere20','spherope'):
                paired=[row for row in rows if row['baseline']==baseline]
                self.assertEqual(len(paired),5);self.assertIn(sum(row['left_method']=='select' for row in paired),(2,3))
        serialized=json.dumps(public)
        self.assertNotIn('sphere20',serialized);self.assertNotIn('spherope',serialized);self.assertNotIn('select',serialized)
        self.assertEqual(human.assignments(self.scenes)[1],public)

    def test_pending_materials_have_no_fake_ratings(self):
        result=human.prepare(self.run,records=[])
        self.assertEqual(result['observed_ratings'],0);self.assertEqual(result['ready_anonymous_assets'],0)
        self.assertEqual(result['ready_assignments'],0);self.assertEqual(len(result['pending_assignment_ids']),240)
        manifest=(self.run/'human/participant/manifest.js').read_text()
        for forbidden in ('sphere20','spherope','select','/home/','/scratch/'):
            self.assertNotIn(forbidden,manifest)
        analyzed=human.analyze(self.run)
        self.assertEqual(analyzed['observations'],0)
        self.assertTrue(all(row['net_preference'] is None and row['ci95'] is None for row in analyzed['results']))
        self.assertEqual(len((self.run/'human/ratings_template.csv').read_text().splitlines()),1)

    def test_unavailable_pair_rejected(self):
        public,unused=self.fixture(False)
        with self.assertRaisesRegex(ValueError,'unavailable'):human.ingest(self.run,self.csv([self.vote(public[0])]))
        self.assertFalse((self.run/'human/ratings/observations.json').exists())

    def test_duplicate_idempotence_and_conflict(self):
        public,unused=self.fixture()
        first=self.vote(public[0]);path=self.csv([first])
        self.assertEqual(human.ingest(self.run,path)['observations'],1)
        self.assertEqual(human.ingest(self.run,path)['observations'],1)
        changed=dict(first,quality='A')
        with self.assertRaisesRegex(ValueError,'Conflicting'):human.ingest(self.run,self.csv([changed],'conflict.csv'))
        self.assertEqual(read(self.run/'human/ratings/observations.json')[0]['quality'],'tie')

    def test_distinct_participants_cannot_share_slot(self):
        public,unused=self.fixture();rows=[row for row in public if row['rater_slot']==1]
        first=self.vote(rows[0]);second=dict(self.vote(rows[1]),participant_id='different-fixture-rater')
        with self.assertRaisesRegex(ValueError,'multiple participants'):human.ingest(self.run,self.csv([first,second]))

    def test_ties_and_crossed_bootstrap_preserve_blinding(self):
        public,private=self.fixture();rows=[]
        for row in public:
            key=private[row['assignment_id']]
            # Two scenes x five independent fixture participants, one comparison.
            if key['prompt_id'] in [self.scenes[0]['prompt_id'],self.scenes[1]['prompt_id']] and key['baseline']=='sphere20':
                choice='tie' if row['rater_slot']==5 else 'A' if key['left_method']=='select' else 'B'
                rows.append(self.vote(row,choice))
        result=human.ingest(self.run,self.csv(rows))
        metric=next(row for row in result['results'] if row['comparison']=='select versus sphere20' and row['criterion']=='quality')
        self.assertEqual((metric['wins'],metric['losses'],metric['ties']),(8,0,2))
        self.assertAlmostEqual(metric['net_preference'],.8);self.assertEqual(metric['bootstrap_replicates'],2000)
        self.assertEqual(metric['missing_ratings'],110);self.assertIsNotNone(metric['ci95'])
        self.assertEqual(human.analyze(self.run)['results'],result['results'])

    def test_viewer_is_offline_and_camera_shared(self):
        script=(Path(human.__file__).parent/'human_assets/viewer.js').read_text()
        self.assertNotIn('fetch(',script);self.assertNotIn('XMLHttpRequest',script)
        self.assertIn('viewers.forEach(v=>v.draw())',script)
        self.assertIn('sequence!==showSequence',script)
        self.assertIn('seam:[180,0]',script);self.assertIn('north:[0,90]',script)

if __name__=='__main__':unittest.main()
