import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split
from torch import nn
from torch.nn import functional as F
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer
from pytorch_pretrained_bert import BertModel
import pandas as pd
import numpy as np
import time
import datetime
import random
import glob
import os
import sys
import joblib
import argparse
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn import metrics
from tqdm import tqdm

USING_GPU = False
DEVICE = None


# https://github.com/malteos/pytorch-bert-document-classification/blob/master/models.py
class ExtraBertMultiClassifier(nn.Module):
    def __init__(self, bert_model_path, labels_count, hidden_dim=768, mlp_dim=100, extras_dim=200, dropout=0.1):
        super().__init__()

        self.config = {
            'bert_model_path': bert_model_path,
            'labels_count': labels_count,
            'hidden_dim': hidden_dim,
            'mlp_dim': mlp_dim,
            'extras_dim': extras_dim,
            'dropout': dropout,
        }

        self.bert = BertModel.from_pretrained(bert_model_path)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + extras_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, labels_count)
        )
        self.softmax = nn.Softmax()

    def forward(self, tokens, masks, extras):
        _, pooled_output = self.bert(
            tokens, attention_mask=masks, output_all_encoded_layers=False)
        dropout_output = self.dropout(pooled_output)

        concat_output = torch.cat((dropout_output, extras), dim=1)
        mlp_output = self.mlp(concat_output.float())
        proba = self.softmax(mlp_output)

        return proba


def format_time(seconds):
    seconds_round = int(round((seconds)))
    return str(datetime.timedelta(seconds=seconds_round))  # hh:mm:ss


def prepare_dataset(sentences, labels, tokenizer, max_length=100):
    input_ids = []
    attention_masks = []
    for sent in sentences:
        # print(sent)
        try:
            encoded_dict = tokenizer.encode_plus(
                sent,
                add_special_tokens=True,
                max_length=max_length,
                truncation=True,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
        except:
            print("some tweet sent is not correct")
            print(sent)
            exit(0)

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    return input_ids, attention_masks, labels

# batch size: 16, 32
# learning rate: 5e-5, 3e-5, 2e-5
# epochs: 2,3,4


def train_bert_model(model, train_dataset, batch_size, epochs=2, learning_rate=2e-5, epsilon=1e-8, extras=False, save_fn=None):
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=batch_size  # Trains with this batch size.
    )
    total_steps = len(train_dataloader) * epochs
    optimizer = AdamW(model.parameters(),
                      lr=learning_rate,
                      eps=epsilon
                      )
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)
    if USING_GPU:
        print("Using GPU", DEVICE)
        model.cuda(DEVICE)

    training_stats = []
    total_t0 = time.time()
    loss_func = torch.nn.CrossEntropyLoss()
    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()
        total_train_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            model.zero_grad()

            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(
                    step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(DEVICE)
            b_input_mask = batch[1].to(DEVICE)
            b_labels = batch[2].to(DEVICE)

            if extras:
                b_extras = batch[3].to(DEVICE)
                b_proba = model(tokens=b_input_ids,
                                masks=b_input_mask,
                                extras=b_extras)

                loss = loss_func(b_proba, b_labels)
                total_train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            else:
                loss, logits, hidden_states = model(b_input_ids,
                                                    token_type_ids=None,
                                                    attention_mask=b_input_mask,
                                                    labels=b_labels)

                total_train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                del loss, logits, hidden_states

        avg_train_loss = total_train_loss / len(train_dataloader)

        training_time = format_time(time.time() - t0)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(
        format_time(time.time()-total_t0)))
    model.eval()

    if save_fn:
        model.save_pretrained(save_fn)
        # model = model_class.from_pretrained('./directory/to/save/')

    return model


def run_bert_model(model, test_dataset, batch_size, extras=False):
    print('Predicting labels for {:,} test sentences...'.format(
        len(test_dataset)))

    if USING_GPU:
        print("Using GPU", DEVICE)
        model.cuda(DEVICE)

    model.eval()
    predictions, true_labels = [], []
    prediction_sampler = SequentialSampler(test_dataset)
    prediction_dataloader = DataLoader(
        test_dataset, sampler=prediction_sampler, batch_size=batch_size)
    for batch in tqdm(prediction_dataloader, total=len(test_dataset)):

        if extras:
            with torch.no_grad():
                b_proba = model(tokens=batch[0].to(DEVICE),
                                masks=batch[1].to(DEVICE),
                                extras=batch[3].to(DEVICE))

                proba = b_proba.detach().cpu().numpy()
                label_ids = batch[2].numpy()

                predictions.append(proba)
                true_labels.append(label_ids)

        else:
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                outputs = model(batch[0].to(DEVICE), token_type_ids=None,
                                attention_mask=batch[1].to(DEVICE))
                b_proba = outputs[0]

                proba = b_proba.detach().cpu().numpy()
                label_ids = batch[2].numpy()

                predictions.append(proba)
                true_labels.append(label_ids)

    print('    DONE.')

    flat_predictions = np.concatenate(predictions, axis=0)
    return flat_predictions
    #y_pred = np.argmax(flat_predictions, axis=1).flatten()

    # return y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, choices=list(range(8)))
    parser.add_argument('--input', type=str, default='..',
                        help='path to the raw input text directory')
    parser.add_argument('--output', type=str, default='../classified-bert/')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for replicability')
    parser.add_argument('--usegraph', action='store_true')
    parser.add_argument('--batchsize', type=int, default=8)
    # parser.add_argument('--logfile', type=str)
    parser.add_argument('--trainedBertModel', type=str)
    args = parser.parse_args()

    if args.output.endswith(".out"):
        print(f"ignore the log file: {args.output}")
        exit(0)

    if os.path.exists(args.output):
        print("{} exists and we pass it".format(args.output))
        exit(0)
    # # ==== parallel running by having the same log file ====
    # # # # define the log and make the parallel
    # LOG_FILE = args.logfile #
    # log_files = []
    # with open(LOG_FILE) as f:
    #     for line in f:
    #         log_files.append(line)
    # if args.output+"\n" in log_files:
    #     print(args.output, "is in the processing and skip it")
    #     exit(0)
    # else:
    #     print(args.output, "is not in the processing and conduct it")
    #     with open(LOG_FILE, 'a') as f:
    #         f.write(args.output+"\n")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    global USING_GPU
    global DEVICE
    if torch.cuda.is_available():
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(args.gpu))
        DEVICE = torch.device("cuda:%s" % args.gpu)
        USING_GPU = True
    else:
        print('No GPU available, using the CPU instead.')
        DEVICE = torch.device("cpu")
        USING_GPU = False

    tag = ""

    df = pd.read_csv(args.input, lineterminator='\n')
    df = df[df['tweet'].notna()]
    X = df.tweet.values
    # RuntimeError: Error(s) in loading state_dict for BertForSequenceClassification:
    # size mismatch for classifier.weight: copying a param with shape torch.Size([3, 768]) from checkpoint, the shape in current model is torch.Size([2, 768]).
    # size mismatch for classifier.bias: copying a param with shape torch.Size([3]) from checkpoint, the shape in current model is torch.Size([2]).
    #
    used_bert_model = args.trainedBertModel
    # # "bert-cross-validation/bert-8-3-5"
    # https://github.com/huggingface/transformers/issues/135
    model = BertForSequenceClassification.from_pretrained(
        used_bert_model, num_labels=3)
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased', do_lower_case=True)
    input_ids, attention_masks, labels = prepare_dataset(
        X, np.zeros(len(X)), tokenizer, max_length=400)
    dataset = TensorDataset(input_ids, attention_masks, labels)

    y_pred = np.zeros(len(X))
    if not args.usegraph:
        flat_logits = run_bert_model(
            model, dataset, batch_size=args.batchsize, extras=False)
        y_pred = np.argmax(flat_logits, axis=1).flatten()
        print(y_pred)

    df['COVID_HATE_BERT_PREDS'] = y_pred
    df.to_csv(args.output)


if __name__ == "__main__":
    main()
