# 清洗数据
# 调参：数据集划分 t5-large或其它大模型 两个文本的MAX_LENGTH BATCH_SIZE EPOCH_NUM 加入decoder_attention_mask lr 选取最优模型并保存 加入scheduler
# generate参数：num_beans, max_length, repetition_penalty, length_penalty, early_stopping, top_k, top_p
# decode参数：clean_up_tokenization_spaces
# 其它遗漏的
# L4/A100
# GPU? TPU?
# 为什么相同的模型消耗的显存比我小？
# 清洗result.csv
# 对比train_dataset里的输出preds与labels

import pandas as pd
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from rouge_score import rouge_scorer
import csv


class MyDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        news = "summarize: " + row[1]
        inputs = tokenizer(news, return_tensors="pt", truncation=True,
                           padding="max_length", max_length=INPUTS_MAX_LENGTH)
        input_ids, attention_mask = inputs['input_ids'].squeeze(
        ), inputs['attention_mask'].squeeze()
        if len(row) == 3:
            summary = row[2]
            labels = tokenizer(summary, return_tensors="pt", truncation=True,
                               padding="max_length", max_length=LABELS_MAX_LENGTH)
            labels = labels['input_ids'].squeeze()
            labels[labels == tokenizer.pad_token_id] = -100
            return input_ids, attention_mask, labels
        else:
            return input_ids, attention_mask


def compute_rouge_l(preds, labels):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    precision_scores = []
    recall_scores = []
    fmeasure_scores = []

    for pred, label in zip(preds, labels):
        score = scorer.score(pred, label)['rougeL']
        precision_scores.append(score.precision)
        recall_scores.append(score.recall)
        fmeasure_scores.append(score.fmeasure)

    rouge_l_p = sum(precision_scores) / len(precision_scores)
    rouge_l_r = sum(recall_scores) / len(recall_scores)
    rouge_l_f = sum(fmeasure_scores) / len(fmeasure_scores)

    return rouge_l_p, rouge_l_r, rouge_l_f


def train(model, train_dataloader, val_dataloader, optimizer, device, NUM_EPOCHS):
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        i = 0
        PRINT_INTERVAL = len(train_dataloader) // 10
        for input_ids, attention_mask, labels in train_dataloader:
            input_ids, attention_mask, labels = input_ids.to(
                device), attention_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            i += 1

            if i % PRINT_INTERVAL == 0:
                print(f"Epoch {epoch}/{i}, Loss: {loss.item()}")

        train_loss = train_loss / len(train_dataloader)
        print(f"Epoch {epoch}, Train Loss: {train_loss}")

        model.eval()
        val_preds = []
        val_labels = []
        val_loss = 0
        with torch.no_grad():
            for input_ids, attention_mask, labels in val_dataloader:
                input_ids, attention_mask, labels = input_ids.to(
                    device), attention_mask.to(device), labels.to(device)
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()

                preds = model.generate(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       max_length=512,
                                       length_penalty=2.0,
                                       num_beams=4,
                                       top_k=50,
                                       top_p=0.95,
                                       do_sample=False,
                                       early_stopping=True,
                                       repetition_penalty=2.5)
                preds = [tokenizer.decode(
                    pred, skip_special_tokens=True) for pred in preds]
                labels[labels == -100] = tokenizer.pad_token_id
                labels = [tokenizer.decode(
                    label, skip_special_tokens=True) for label in labels]

                val_preds.extend(preds)
                val_labels.extend(labels)

        val_loss = val_loss / len(val_dataloader)
        rouge_l_p, rouge_l_r, rouge_l_f = compute_rouge_l(
            val_preds, val_labels)
        print(
            f"Epoch {epoch}, Validation Loss: {val_loss}, ROUGE-L P: {rouge_l_p}, R: {rouge_l_r}, F: {rouge_l_f}")


def test(model, test_dataloader):
    model.eval()
    test_preds = []
    with torch.no_grad():
        for input_ids, attention_mask in test_dataloader:
            input_ids, attention_mask = input_ids.to(
                device), attention_mask.to(device)
            preds = model.generate(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   max_length=512,
                                   length_penalty=2.0,
                                   num_beams=4,
                                   top_k=50,
                                   top_p=0.95,
                                   do_sample=False,
                                   early_stopping=True,
                                   repetition_penalty=2.5)
            preds = [tokenizer.decode(
                pred, skip_special_tokens=True, clean_up_tokenization_spaces=True) for pred in preds]
            test_preds.extend(preds)

    print(test_preds[0])
    print(len(test_preds))
    return test_preds


if __name__ == "__main__":
    train_val_df = pd.read_csv(
        "train_dataset.csv", delimiter="\t", header=None)
    test_df = pd.read_csv("test_dataset.csv", delimiter="\t", header=None)

    INPUTS_MAX_LENGTH = 1024
    LABELS_MAX_LENGTH = 128
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 2

    tokenizer = PegasusTokenizer.from_pretrained(
        "google/pegasus-cnn_dailymail")
    model = PegasusForConditionalGeneration.from_pretrained(
        "google/pegasus-cnn_dailymail")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    train_val_dataset = MyDataset(train_val_df)
    test_dataset = MyDataset(test_df)
    train_dataset, val_dataset = random_split(train_val_dataset, [0.9, 0.1])
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    train(model, train_dataloader, val_dataloader,
          optimizer, device, NUM_EPOCHS)
    test_preds = test(model, test_dataloader)

    with open("result_wc_0.csv", mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')

        for index, pred in enumerate(test_preds):
            writer.writerow([index, pred])
