"""Offline checks for archive containment, cohort identity, and caption schema."""
import hashlib
import json
import unittest

from .dataset import parse_listing, safe_member, select_rows
from .captions import validate_caption, five_bytes, payload


class ArchiveTests(unittest.TestCase):
    def test_reject_path_traversal_and_duplicate_names(self):
        for value in ('../escape.jpg','/absolute.jpg','a/../../escape','C:\\escape.jpg','a\nb.jpg'):
            with self.assertRaises(ValueError):safe_member(value)
        self.assertEqual(safe_member('lau_dataset\\odisr\\testing\\HR\\001.jpg'),
                         'lau_dataset/odisr/testing/HR/001.jpg')
        listing='Path = a.jpg\nSize = 10\nCRC = 12345678\n\nPath = a.jpg\nSize = 10\n'
        with self.assertRaises(ValueError):parse_listing(listing)

    def test_listing_distinguishes_files_and_folders(self):
        listing='Path = folder\nFolder = +\nSize = 0\n\nPath = folder/a.jpg\nSize = 12\nCRC = ABCDEF01\nEncrypted = -\n'
        self.assertEqual(parse_listing(listing),[dict(member='folder/a.jpg',bytes=12,crc32='ABCDEF01',encrypted='-',attributes=None,method=None,solid=None)])


class SelectionTests(unittest.TestCase):
    def fixture(self):
        rows=[];annotations=[]
        for i in range(32):
            member='lau_dataset/odisr/testing/HR/%03d.jpg'%i
            rows.append(dict(member=member,index=i,image_sha256='image%d'%i,pixel_sha256='pixels%d'%i,selection_key='%03d'%i))
            annotations.append(dict(member=member,index=i,image_sha256='image%d'%i,
                category='indoor' if i<10 else 'outdoor',scene_group='same' if i in (0,1) else str(i),excluded_reason=None))
        return dict(candidates=rows),annotations

    def test_fixed_counts_determinism_and_duplicate_scene_exclusion(self):
        review,annotations=self.fixture()
        first=select_rows(review,annotations);second=select_rows(review,list(reversed(annotations)))
        self.assertEqual(first,second)
        self.assertEqual(sum(r['category']=='indoor' for r in first),6)
        self.assertEqual(sum(r['category']=='outdoor' for r in first),18)
        self.assertEqual(sum(r['second_seed'] for r in first),8)
        self.assertEqual(len({r['scene_group'] for r in first}),24)
        self.assertNotIn(1,[r['index'] for r in first])

    def test_insufficient_categories_and_tampered_reviews_fail(self):
        review,annotations=self.fixture();annotations[3]['image_sha256']='changed'
        with self.assertRaises(ValueError):select_rows(review,annotations)
        review,annotations=self.fixture()
        for row in annotations:row['category']='outdoor'
        with self.assertRaises(ValueError):select_rows(review,annotations)


class CaptionTests(unittest.TestCase):
    def fixture(self):
        return dict(global_generation_description='A quiet room with pale walls and a wooden floor.',
            directional_descriptions=['A white ceiling with a round light.','Pale upper walls and a window.','A wooden desk stands beside the window.',
                                      'A chair rests beside the wooden desk.','Brown wooden floorboards below.'],
            scoring_descriptions=['White ceiling and round light.','Pale walls with window.','Wooden desk by window.',
                                  'Chair beside desk.','Brown floorboards.'],uncertainties='None')

    def test_five_lines_exact_bytes_and_distinct_scoring_namespace(self):
        record=validate_caption(json.dumps(self.fixture()))
        raw=five_bytes(record['directional_descriptions']);short=five_bytes(record['scoring_descriptions'])
        self.assertEqual(raw.count(b'\n'),5)
        self.assertTrue(raw.endswith(b'\n'))
        self.assertNotEqual(hashlib.sha256(raw).hexdigest(),hashlib.sha256(short).hexdigest())

    def test_duplicate_bands_newlines_and_excess_words_rejected(self):
        for mode in ('duplicate','newline','long','extra'):
            record=self.fixture()
            if mode=='duplicate':record['directional_descriptions'][1]=record['directional_descriptions'][0]
            elif mode=='newline':record['directional_descriptions'][1]+='\nextra'
            elif mode=='long':record['scoring_descriptions'][0]=' '.join(['word']*36)
            else:record['quality_score']=5
            with self.assertRaises(ValueError):validate_caption(record)

    def test_payload_preserves_image_and_frozen_provider(self):
        import base64
        frozen=dict(judge=dict(model='openai/gpt-4o',provider=dict(only=['openai'],allow_fallbacks=False),
            output_format='strict_json_text',parameters=dict(max_tokens=2200,temperature=0)),rubric='Describe visible facts.',schema={})
        result=payload(b'actual image bytes',frozen)
        encoded=result['messages'][0]['content'][1]['image_url']['url'].split(',',1)[1]
        self.assertEqual(base64.b64decode(encoded),b'actual image bytes')
        self.assertFalse(result['provider']['allow_fallbacks'])
        self.assertEqual(result['max_tokens'],2200)


if __name__=='__main__':unittest.main()
