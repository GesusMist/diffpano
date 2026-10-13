from studies.ournew_comparison.common import *
from studies.ournew_comparison.official import original
from studies.ournew_comparison.validate import result

def main():
    gate('cpu.json');gate('weights.json');m=freeze_check();row=m['rows'][0]
    _,path=prompt(row['prompt']);image,record=original(row,path,preflight=True)
    assert image is None and record['schedule']==m['official_reference']['schedule']
    assert record['initialization']['initial_state_sha256']==m['official_reference']['initialization']['initial_state_sha256']
    result('official.json',passed=True,record=record,extra_full_trajectories=0)
if __name__=='__main__':main()
