import json
import torch
import torch.nn as nn
import logging
import collections
from util import AverageMeter


def predict(model, test_loader, args):
    model.eval()
    predicted = {}

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            sample_id = batch['sample_id'].tolist()
            question_features = batch['question_features']
            attended_question_features = batch['attended_question_features']
            answer_features = batch['answer_features']
            transcript_features = batch['transcript_features']
            transcript_attention_mask = batch['transcript_attention_mask']
            q_q_a_encoder_attention_mask =  batch['q_q_a_encoder_attention_mask']
            
            model_output = model(
                question_features, 
                attended_question_features, 
                answer_features, 
                transcript_features, 
                transcript_attention_mask,
                q_q_a_encoder_attention_mask,
            )
            predicted.update(zip(sample_id, torch.max(model_output.logits, dim=1).indices.cpu().tolist()))

    with open(args.predict_path, 'w') as predict_file:
        json.dump({'predictions': predicted}, predict_file)
    logging.info(f"Prediction saved in {args.predict_path}")


def eval(model, val_loader, args, test=False):
    model.eval()
    count = 0
    metrics = collections.defaultdict(int)

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            question_features = batch['question_features']
            attended_question_features = batch['attended_question_features']
            answer_features = batch['answer_features']
            transcript_features = batch['transcript_features']
            transcript_attention_mask = batch['transcript_attention_mask']
            q_q_a_encoder_attention_mask =  batch['q_q_a_encoder_attention_mask']
            answer_id = batch['answer_id']
            count += answer_id.size(0)
            
            model_output = model(
                question_features, 
                attended_question_features, 
                answer_features, 
                transcript_features, 
                transcript_attention_mask,
                q_q_a_encoder_attention_mask,
                answer_id=answer_id
            )

            predicted = torch.max(model_output.logits, dim=1).indices.cpu()
            metrics["acc"] += (predicted == answer_id).sum().item()

    step = "val" if not test else "test"
    for k in metrics:
        v = metrics[k] / count
        logging.info(f"{step} {k}: {v:.2%}")

    return metrics["acc"] / count


def train(model, train_loader, optimizer, criterion, scheduler, epoch, args):
    model.train()
    running_loss, running_acc = (
        AverageMeter(),
        AverageMeter(),
    )
    for i, batch in enumerate(train_loader):
        question_features = batch['question_features']
        attended_question_features = batch['attended_question_features']
        answer_features = batch['answer_features']
        transcript_features = batch['transcript_features']
        transcript_attention_mask = batch['transcript_attention_mask']
        q_q_a_encoder_attention_mask =  batch['q_q_a_encoder_attention_mask']
        answer_id = batch['answer_id']

        model_output = model(
            question_features, 
            attended_question_features, 
            answer_features, 
            transcript_features, 
            transcript_attention_mask,
            q_q_a_encoder_attention_mask,
            answer_id=answer_id
        )
        
        N = answer_id.size(0)
        predicted = torch.max(model_output.logits, dim=1).indices.cpu()
        # print(model_output.logits, predicted, answer_id)
        running_acc.update((predicted == answer_id).sum().item() / N, N)

        optimizer.zero_grad()
        model_output.loss.backward()
        if args.clip:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
        optimizer.step()
        scheduler.step()
        
        running_loss.update(model_output.loss.detach().cpu().item(), N)
        if (i + 1) % (len(train_loader) // args.freq_display) == 0:
            logging.info(
                f"Epoch {epoch + 1}, Epoch status: {float(i + 1) / len(train_loader):.4f}, Training VideoQA loss: "
                f"{running_loss.avg:.4f}, Training acc: {running_acc.avg:.2%}"
            )
    running_acc.reset()
    running_loss.reset()
