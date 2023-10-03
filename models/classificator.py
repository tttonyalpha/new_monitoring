from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, accuracy_score
from transformers import TrainingArguments, Trainer


class dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length, summarize_long_sentence):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.summarize_long_sentence = summarize_long_sentence

    def __getitem__(self, index):
        text = str(self.dataframe.sentence[index])
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_length and self.summarize_long_sentence:
            text = summarize(text)

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'labels': torch.tensor(self.dataframe.label[index], dtype=torch.long)
        }

    def __len__(self):
        return len(self.dataframe)


class test_dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, index):
        text = str(self.dataframe.summary[index])
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long)
        }

    def __len__(self):
        return len(self.dataframe)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds)
    return {'F1': f1}


def predict(data_loader, model):
    predictions = []
    probs = []
    pbar = tqdm(data_loader)
    for i, batch in enumerate(pbar, 1):
        with torch.no_grad():
            outputs = model(batch['input_ids'].to(
                device), attention_mask=batch['attention_mask'].to(device))
            logits = outputs.logits
            prob = torch.nn.functional.softmax(
                logits, dim=1)[:, 1:].detach().cpu().numpy().flatten()
            logits = logits.detach().cpu().numpy()
            pred = np.argmax(logits, axis=1).flatten()

        predictions += list(pred)
        probs += list(prob)
    return predictions, probs
