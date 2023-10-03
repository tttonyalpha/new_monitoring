from tqdm.auto import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "IlyaGusev/mbart_ru_sum_gazeta"
sum_tokenizer = MBartTokenizer.from_pretrained(model_name)
sum_model = MBartForConditionalGeneration.from_pretrained(
    model_name, torch_dtype=torch.float16)

sum_model = BetterTransformer.transform(sum_model)
sum_model.to(device)
sum_model.eval()

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def get_summary(data_loader):
    # batched summarization
    summarized = []

    pbar = tqdm(data_loader, desc=f'Summarizing')
    for i, batch in enumerate(pbar, 1):
        encoding = sum_tokenizer(batch, padding='max_length',
                                 truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            generated_ids = sum_model.generate(
                **encoding, no_repeat_ngram_size=4, max_new_tokens=100)
        summarized += sum_tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)
    return summarized


def summarize(text, max_input_length=1024, max_output_length=512):
    # dummy summarization for debug
    input_ids = sum_tokenizer(
        [text],
        max_length=max_input_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )["input_ids"].to(device)

    output_ids = sum_model.generate(
        input_ids=input_ids,
        no_repeat_ngram_size=4,
        max_length=max_output_length
    )[0]

    summary = sum_tokenizer.decode(output_ids, skip_special_tokens=True)
    return summary
