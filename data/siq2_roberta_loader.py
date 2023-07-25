import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.dataloader import default_collate
import pandas as pd


class SIQ2RoBERTaDataset(Dataset):
    def __init__(
        self,
        csv_path,
        questions_features,
        attended_questions_features,
        answers_features,
        transcripts_features,
        skip_transcript_prob=0.0,
        load_answer_id=True
    ):
        """
        :param csv_path: path to a csv containing columns video_id, question, answer
        :param questions_features: dictionary mapping index to torch tensor of features of questions
        :param attended_questions_features: dictionary mapping index to torch tensor of features of attended questions
        :param answers_features: dictionary mapping index to torch tensor of features of answers
        :param transcripts_features: dictionary mapping video_id to torch tensor of features of its transcript
        :param load_answer_id: load the answers or not (can be used at inference time)
        """
        self.data = pd.read_csv(csv_path, keep_default_na=False)
        self.questions_features = questions_features
        self.attended_questions_features = attended_questions_features
        self.answers_features = answers_features
        self.transcripts_features = transcripts_features
        self.load_answer_id = load_answer_id
        self.skip_transcript_prob = skip_transcript_prob

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample_id = self.data["sample_id"].values[index]
        question_id = self.data["qid"].values[index]
        vid_id = self.data["video_id"].values[index]
        question_features = self.questions_features[sample_id]
        attended_question_features = self.attended_questions_features[sample_id]
        answer_features = self.answers_features[sample_id]
        q_q_a_encoder_attention_mask = torch.ones(6) # attended_questions + 4 answers + attended_question_transcript
        if vid_id in self.transcripts_features and torch.rand(1) >= self.skip_transcript_prob:
            transcript_features = self.transcripts_features[vid_id]
        else:
            transcript_features = torch.zeros(1, 768).cuda()
            q_q_a_encoder_attention_mask[-1] = 0

        if self.load_answer_id:
            answer_id = int(self.data["answer"].values[index])

        if self.load_answer_id:
            return {
                "sample_id": sample_id,
                "question_id": question_id,
                "video_id": vid_id,
                "question_features": question_features,
                "attended_question_features": attended_question_features,
                "answer_features": answer_features,
                "transcript_features": transcript_features,
                "q_q_a_encoder_attention_mask": q_q_a_encoder_attention_mask,
                "answer_id": answer_id,
            }
        else:
            return {
                "sample_id": sample_id,
                "question_id": question_id,
                "video_id": vid_id,
                "question_features": question_features,
                "attended_question_features": attended_question_features,
                "answer_features": answer_features,
                "transcript_features": transcript_features,
                "q_q_a_encoder_attention_mask": q_q_a_encoder_attention_mask,
            }


def siq2_roberta_collate_fn(batch):
    """
    :param batch: [dataset[i] for i in N]
    :return: batch padded to the max length of the batch
    """
    transcripts_dim0_max_len = max(batch[i]["transcript_features"].shape[0] for i in range(len(batch)))
    for i in range(len(batch)):
        if batch[i]["transcript_features"].shape[0] < transcripts_dim0_max_len:
            batch[i]["transcript_features"] = torch.cat(
                [
                    batch[i]["transcript_features"],
                    torch.zeros(
                        transcripts_dim0_max_len - batch[i]["transcript_features"].shape[0], 
                        batch[i]["transcript_features"].shape[1],
                    ).cuda(),
                ],
                0,
            )
            batch[i]["transcript_attention_mask"] = torch.cat(
                [
                    torch.ones(
                        batch[i]["transcript_features"].shape[0]
                    ).cuda(),
                    torch.zeros(
                        transcripts_dim0_max_len - batch[i]["transcript_features"].shape[0]
                    ).cuda(),
                ],
                0,
            )
        else: # == transcripts_dim0_max_len
            batch[i]["transcript_attention_mask"] = torch.cat(
                [
                    torch.ones(
                        batch[i]["transcript_features"].shape[0]
                    ).cuda()
                ],
                0,
            )
        batch[i]["transcript_attention_mask"] = batch[i]["transcript_attention_mask"] == 0 # Make it bool

    return default_collate(batch)


def get_siq2_roberta_loaders(args, questions_features, attended_questions_features, answers_features, transcripts_features):
    test_dataset = SIQ2RoBERTaDataset(
        csv_path=args.test_csv_path,
        questions_features=questions_features['test'],
        attended_questions_features=attended_questions_features['test'],
        answers_features=answers_features['test'],
        transcripts_features=transcripts_features,
        skip_transcript_prob=args.skip_transcript_prob,
        load_answer_id=args.load_answer_id
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        collate_fn=siq2_roberta_collate_fn,
    )

    val_dataset = SIQ2RoBERTaDataset(
        csv_path=args.val_csv_path,
        questions_features=questions_features['val'],
        attended_questions_features=attended_questions_features['val'],
        answers_features=answers_features['val'],
        transcripts_features=transcripts_features,
        skip_transcript_prob=args.skip_transcript_prob,
        load_answer_id=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        collate_fn=siq2_roberta_collate_fn,
    )

    train_dataset = SIQ2RoBERTaDataset(
        csv_path=args.train_csv_path,
        questions_features=questions_features['train'],
        attended_questions_features=attended_questions_features['train'],
        answers_features=answers_features['train'],
        transcripts_features=transcripts_features,
        skip_transcript_prob=args.skip_transcript_prob,
        load_answer_id=True
    )

    if args.use_validation:
        train_loader = DataLoader(
            ConcatDataset([train_dataset, val_dataset]),
            batch_size=args.batch_size,
            num_workers=args.num_thread_reader,
            shuffle=args.load_answer_id, # Only on training not feature extraction
            collate_fn=siq2_roberta_collate_fn,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_thread_reader,
            shuffle=args.load_answer_id, # Only on training not feature extraction
            collate_fn=siq2_roberta_collate_fn,
        )
    return (
        train_dataset,
        train_loader,
        val_dataset,
        val_loader,
        test_dataset,
        test_loader,
    )
