"""Run exactly one approved additional FLUX prompt/mode pair."""
import argparse
from studies.gradient_blending.common import *
from studies.gradient_blending.run import run

def main():
    assert BACKEND=='flux' and SUITE=='flux-scenes20'
    rows=scene_rows();p=argparse.ArgumentParser();p.add_argument('--index',type=int,choices=range(len(rows)),required=True)
    args=p.parse_args();row=rows[args.index]
    assert row in read(OUT/'manifest.json')['rows'] and row['prompt'] in NEW_FLUX_PROMPTS
    run(row['prompt'],row['mode'])
if __name__=='__main__':main()
