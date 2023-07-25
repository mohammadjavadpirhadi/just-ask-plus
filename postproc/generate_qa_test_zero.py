import pandas as pd
import json

import os
import sys
sys.path.insert(0, os.getcwd())  # to correct with parent folder
from args import get_args
from global_parameters import SIQ2_PATH

os.chdir(SIQ2_PATH)

args = get_args()

with open(os.path.join(args.checkpoint_dir, 'predict.json')) as predicts_file:
    predicts = json.load(predicts_file)
predictions = {'sample_id': list(predicts['predictions'].keys()), 'answer_idx': list(predicts['predictions'].values())}
predictions_df = pd.DataFrame.from_dict(predictions)
predictions_df['sample_id']=predictions_df['sample_id'].astype(int)

test_df = pd.read_csv('test.csv', keep_default_na=False)
test_df = test_df.join(predictions_df.set_index('sample_id'), on='sample_id')
test_df.rename(columns={
    "video_id": "vid_name",
    "question": "q",
    "a1": "a0",
    "a2": "a1",
    "a3": "a2",
    "a4": "a3"
    },
    inplace=True
)
test_df.drop(columns=['sample_id', 'Unnamed: 0'], inplace=True)

with open('qa_test_zero.json', 'w') as qa_test_zero_file:
    qa_test_zero_file.write(test_df.to_json(orient='records', lines=True))
