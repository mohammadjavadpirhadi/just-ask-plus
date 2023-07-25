# [Social-IQ 2.0 Challenge @ ICCV 2023] Just Ask Plus: Using Transcripts for VideoQA

**This repository is a copy of the [Just Ask](https://github.com/antoyang/just-ask) repository. We provide only the required parts for reproducing our results of the Social-IQ 2.0 Challenge here**

## Download the dataset

You can clone the official repository and download the dataset yourself or if you wish to reproduce exactly the same results, you can access the version we used [here](https://drive.google.com/drive/folders/1ELWrTTH5OA2y1pMD1iCopv-VFjxgx3GZ?usp=sharing).


## Paths and Requirements
Fill the empty paths in the file `global_parameters.py`.

To install requirements, run:
```
pip install -r requirements.txt
```

## Extract video features
We provide in the `extract` folder the code to extract features with the S3D feature extractor. It requires downloading the S3D model weights available at [this repository](https://github.com/antoine77340/S3D_HowTo100M). The `s3d_howto100m.pth` checkpoint and `s3d_dict.npy` dictionary should be in `DEFAULT_MODEL_DIR`.

**Extraction**: You should prepare for each dataset a csv with columns `video_path` (typically in the form of *<dataset_path>/video/<video_path>*), and `feature_path` (typically in the form of *<dataset_path>/features/<video_path>.npy*). Then use (you may launch this script on multiple GPUs to fasten the extraction process):
```
python extract/extract.py --csv <csv_path>
```

**Merging**: To merge the extracted features into a single file for each VideoQA dataset, use (for ActivityNet-QA that contains long videos, add `--pad 120`):
```
python extract/merge_features.py --folder <features_path> \
--output_path <DEFAULT_DATASET_DIR>/s3d.pth --dataset <dataset>
```


## Preprocess the dataset

Run the following command:

```
python preproc/preproc_siq2.py
```

## Evaluate checkpoint for zero-shot settings

You should download the intended checkpoint (you can find them [here](https://github.com/antoyang/just-ask)) and provide the path with ```--pretrain_path``` and ```--zeroshot_eval=1``` flags.

For example you can evaluate the Just Ask model using [HowToVQA69M + WebVidVQA3M + How2QA checkpoint](https://drive.google.com/file/d/1ITlnINPMBP5dTabPxUG-3o5uxdoUW7DY/view?usp=sharing) by following command:

```
python main_videoqa.py \
--checkpoint_dir=<checkpoint_dir> \
--dataset=siq2 \
--pretrain_path=ckpt_pt_howtovqa69m.pth \
--zeroshot_eval=1
```

## Extract question and suggested answer features

We used the best checkpoint of the previous part which is [HowToVQA69M + WebVidVQA3M + How2QA](https://drive.google.com/file/d/1ITlnINPMBP5dTabPxUG-3o5uxdoUW7DY/view?usp=sharing) to extract the features. To do that download the checkpoint and run the following command:

```
python main_videoqa.py \
--checkpoint_dir=<checkpoint_dir> \
--dataset=siq2 \
--pretrain_path=ckpt_ft2_how2qa.pth \
--save_questions_feature \
--save_attended_questions_feature \
--save_answers_feature
```

## Extract transcript features

Run the following command to extract transcript features using SpeechT5:

```
python extract/extract_transcripts_speecht5.py
```

To use the RoBERTa-base instead of SpeechT5 simply replace ```extract_transcripts.py``` with ```extract_transcripts_roberta.py```.

## Train the model

We set the epochs to 15 but the best-performing epoch on the validation set is saved as the ```best_model.pth``` in ```<checkpoint_dir>```. Use the following command to train the model:

```
python main_siq2.py \
--checkpoint_dir=<checkpoint_dir> \
--dataset=siq2 \
--skip_transcript_prob=<p> \
--epochs=15 \
--siq2_questions_features_path=<path_to_questions_features.pth> \
--siq2_attended_questions_features_path=<path_to_attended_questions_features.pth> \
--siq2_answers_features_path=<path_to_answers_features.pth> \
--siq2_transcripts_features_path=<path_to_transcript_sentences_features.pth>
```

**To use the validation set for training too use the ```--use_validation=1``` flag.**

**If you have problems loading the dataset use the```--num_thread_reader=0``` flag.**

To use the RoBERTa-base instead of SpeechT5 simply replace ```main_siq2.py``` with ```main_siq2_roberta.py```.

## Prediction

To predict the answers in a zero-shot manner, run the following command:

```
python just-ask/main_videoqa.py \
--checkpoint_dir=<checkpoint_dir> \
--dataset=siq2 \
--pretrain_path=ckpt_ft2_how2qa.pth \
--predict=1
```

and to use the trained model:

```
python just-ask/main_siq2.py \
--checkpoint_dir=<checkpoint_dir> \
--dataset=siq2 \
--skip_transcript_prob=<p> \
--pretrain_path=<path_to_best_model.py> \
--predict=1 \
--siq2_questions_features_path=<path_to_questions_features.pth> \
--siq2_attended_questions_features_path=<path_to_attended_questions_features.pth> \
--siq2_answers_features_path=<path_to_answers_features.pth> \
--siq2_transcripts_features_path=<path_to_transcript_sentences_features.pth>
```

These commands save a file named ```predict.json``` in ```<checkpoint_dir>``` which maps each sample id (added in the preprocessing step) to the predicted answer.

## Generate qa_test_{focus}

To generate the submission file for zero-shot focus run:

```
python postproc/generate_qa_test_zero.py \
--checkpoint_dir=<checkpoint_dir> \
```

and for fusion and reasoning focus run:

```
python postproc/generate_qa_test_fusion.py \
--checkpoint_dir=<checkpoint_dir> \
```

The ```<checkpoint_dir>``` must contain ```predict.json```.

