import json
import jsonlines
import pandas as pd

import os
import sys
sys.path.insert(0, os.getcwd())  # to correct with parent folder
from global_parameters import SIQ2_PATH

os.chdir(SIQ2_PATH)

with open('current_split.json') as json_file:
    current_split = json.load(json_file)

videos_id = []
for subset in ['youtubeclips', 'movieclips', 'car']:
    for split in ['train', 'val', 'test']:
        videos_id += current_split['subsets'][subset][split]

missing_videos = {
    'train': [],
    'val': [],
    'test': []
}

for split in ['train', 'val', 'test']:
    with jsonlines.open(f'qa/qa_{split}.json') as json_file:
        qa_records = []
        for qa in json_file:
            if qa['vid_name'] in videos_id:
                qa_records.append(qa)
            elif qa['vid_name'] not in missing_videos[split]:
                missing_videos[split].append(qa['vid_name'])
        qa_df = pd.DataFrame.from_records(qa_records)
        sample_id = list(range(len(qa_df)))
        qa_df['sample_id'] = sample_id
        qa_df.rename(columns={
            "vid_name": "video_id",
            "answer_idx": "answer",
            "q": "question",
            "a0": "a1",
            "a1": "a2",
            "a2": "a3",
            "a3": "a4"
            },
            inplace=True
        )
        qa_df.to_csv(f'{split}.csv')

with open('missing_videos.json', 'w') as missing_videos_file:
    json.dump(missing_videos, missing_videos_file)
