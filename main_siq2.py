import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import logging
from transformers import get_cosine_schedule_with_warmup
from args import get_args
from data.siq2_loader import get_siq2_loaders
from model.just_ask_plus_model import JustAskPlusConfig, JustAskPlusModel
from train.train_just_ask_plus import train, eval, predict


# args, logging
args = get_args()
if not (os.path.isdir(args.save_dir)):
    os.mkdir(os.path.join(args.save_dir))
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
)
logFormatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
rootLogger = logging.getLogger()
fileHandler = logging.FileHandler(os.path.join(args.save_dir, "stdout.log"), "w+")
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)
logging.info(args)

# set random seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)


# Model
config = JustAskPlusConfig()
model = JustAskPlusModel(config=config)
model.cuda()
logging.info("Using {} GPUs".format(torch.cuda.device_count()))

# Load pretrain path
model = nn.DataParallel(model)
if args.pretrain_path != "":
    model.load_state_dict(torch.load(args.pretrain_path))
    logging.info(f"Loaded checkpoint {args.pretrain_path}")
logging.info(
    f"Nb of trainable params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}"
)

# Dataloaders
questions_features = torch.load(args.siq2_questions_features_path, map_location='cuda')
attended_questions_features = torch.load(args.siq2_attended_questions_features_path, map_location='cuda')
answers_features = torch.load(args.siq2_answers_features_path, map_location='cuda')
transcripts_features = torch.load(args.siq2_transcripts_features_path, map_location='cuda')
(
    train_dataset,
    train_loader,
    val_dataset,
    val_loader,
    test_dataset,
    test_loader,
) = get_siq2_loaders(args, questions_features, attended_questions_features, answers_features, transcripts_features)

logging.info("number of train instances: {}".format(len(train_loader.dataset)))
logging.info("number of val instances: {}".format(len(val_loader.dataset)))
logging.info("number of test instances: {}".format(len(test_loader.dataset)))

# DO NOT FORGET
criterion = nn.CrossEntropyLoss()
params_for_optimization = list(p for p in model.parameters() if p.requires_grad)
optimizer = optim.Adam(
    params_for_optimization, lr=args.lr, weight_decay=args.weight_decay
)
criterion.cuda()

if not args.zeroshot_eval and not args.test and not args.predict and args.load_answer_id:
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 0, len(train_loader) * args.epochs
    )
    logging.info(
        f"Set cosine schedule with {len(train_loader) * args.epochs} iterations"
    )
    if not args.use_validation:
        eval(model, val_loader, args, False)  # zero-shot
        best_val_acc = -float("inf")
        best_epoch = 0
    for epoch in range(args.epochs):
        train(model, train_loader, optimizer, criterion, scheduler, epoch, args)
        if not args.use_validation:
            val_acc = eval(model, val_loader, args, False)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(
                    model.state_dict(), os.path.join(args.save_dir, "best_model.pth")
                )
    if not args.use_validation:
        logging.info(f"Best val model at epoch {best_epoch + 1}")
        model.load_state_dict(
            torch.load(
                os.path.join(args.checkpoint_predir, args.checkpoint_dir, "best_model.pth")
            )
        )
    else:
        torch.save(
            model.state_dict(), os.path.join(args.save_dir, "best_model.pth")
        )

if args.zeroshot_eval:
    eval(model, val_loader, args, False)

if args.test:
    # Evaluate on test set
    eval(model, test_loader, args, True)

if args.predict:
    predict(model, test_loader, args)
