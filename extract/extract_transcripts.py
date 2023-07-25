import nltk
import json
import gc
from tqdm import tqdm
import webvtt
from nemo.collections import nlp as nemo_nlp
from nltk.tokenize import sent_tokenize
from transformers import SpeechT5Processor
from transformers.models.speecht5.modeling_speecht5 import SpeechT5EncoderWithTextPrenet
import torch

import os
import sys
sys.path.insert(0, os.getcwd())  # to correct with parent folder
from global_parameters import SIQ2_PATH

os.chdir(SIQ2_PATH)

nltk.download('punkt')

with open('current_split.json') as json_file:
    current_split = json.load(json_file)

videos_id = []
for subset in ['youtubeclips', 'movieclips', 'car']:
    for split in ['train', 'val', 'test']:
        videos_id += current_split['subsets'][subset][split]

print(f'Found {len(videos_id)} videos.')

pretrained_model = nemo_nlp.models.PunctuationCapitalizationModel.from_pretrained("punctuation_en_bert")
speecht5_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
speecht5_text_encoder = SpeechT5EncoderWithTextPrenet.from_pretrained("microsoft/speecht5_tts")

if not os.path.isdir("transcript_sentences_features/"):
    os.mkdir("transcript_sentences_features/")

empty_transcripts = []

for video_id in tqdm(videos_id):
    if os.path.isfile(f"transcript_sentences_features/{video_id}_f.pth") and \
    os.path.isfile(f"transcript_sentences_features/{video_id}_am.pth"):
        continue

    transcript_path = f'transcript/{video_id}.vtt'
    transcript = webvtt.read(transcript_path)

    caption_lines = []
    for caption in transcript:
        if caption.text.strip():
            caption_lines += caption.text.strip().split('\n')

    deduplicated_caption_lines = []
    for caption_line in caption_lines:
        if len(deduplicated_caption_lines) == 0 or deduplicated_caption_lines[-1] != caption_line:
            deduplicated_caption_lines.append(caption_line)

    clean_transcript = pretrained_model.add_punctuation_capitalization(
        [' '.join(deduplicated_caption_lines)],
        max_seq_length=128,
        step=8,
        margin=16,
        batch_size=32,
    )
    assert len(clean_transcript) == 1

    transcript_sentences = sent_tokenize(clean_transcript[0])
    if not len(transcript_sentences):
        empty_transcripts.append(video_id)
        gc.collect()
        continue

    speecht5_text_inputs = speecht5_processor(text=transcript_sentences, padding=True, truncation=True, max_length=450, return_tensors="pt")
    speecht5_text_inputs['input_values'] = speecht5_text_inputs.pop('input_ids')
    transcript_sentences_features = speecht5_text_encoder(**speecht5_text_inputs).last_hidden_state
    assert transcript_sentences_features.shape[0] == len(transcript_sentences)

    torch.save(transcript_sentences_features, f"transcript_sentences_features/{video_id}_f.pth")
    torch.save(speecht5_text_inputs['attention_mask'], f"transcript_sentences_features/{video_id}_am.pth")

    gc.collect()

with open('empty_transcripts.txt', 'w') as empty_transcripts_file:
    for empty_transcript in empty_transcripts:
        empty_transcripts_file.write(f'{empty_transcript}\n')
print("empty_transcripts:", empty_transcripts)

gc.collect()

transcript_sentences_features = {}
for video_id in tqdm(videos_id):
    if video_id in empty_transcripts:
        continue
    features = torch.load(f'transcript_sentences_features/{video_id}_f.pth', map_location=torch.device('cpu'))
    features.requires_grad = False
    attention_mask = torch.load(f'transcript_sentences_features/{video_id}_am.pth', map_location=torch.device('cpu'))
    attention_mask.requires_grad = False
    transcript_sentences_features[video_id] = {
        'features': features,
        'attention_mask': attention_mask
    }

torch.save(transcript_sentences_features, 'transcript_sentences_features.pth')
