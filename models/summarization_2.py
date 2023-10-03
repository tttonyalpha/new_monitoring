from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm.auto import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "cointegrated/rut5-base-absum"
sum_tokenizer = T5Tokenizer.from_pretrained(model_name)
sum_model = T5ForConditionalGeneration.from_pretrained(
    model_name, torch_dtype=torch.float16)

sum_model = BetterTransformer.transform(sum_model)
sum_model.to(device)
sum_model.eval()

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def summarize(data_loader, test_words_distr):
    summarized = []
    words_nums = np.sort(np.array(test_words_distr))

    pbar = tqdm(test_sorted_loader, desc=f'Summarizing')

    for batch, n_words in zip(pbar, words_nums):
        encoding = sum_tokenizer(
            batch, padding=True, truncation=False, return_tensors='pt').to(device)
        with torch.no_grad():
            generated_ids = sum_model.generate(
                **encoding, max_new_tokens=120, repetition_penalty=10.0)
        summarized += sum_tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)

    return summarized


def summarize(
    text, n_words=None, compression=None,
    max_length=1000, num_beams=3, do_sample=False, repetition_penalty=10.0,
    **kwargs
):
    # dummy sum
    if n_words:
        text = '[{}] '.format(n_words) + text
    elif compression:
        text = '[{0:.1g}] '.format(compression) + text
    print(text)
    x = sum_tokenizer(text, return_tensors='pt', padding=True).to(device)
    with torch.inference_mode():
        out = sum_model.generate(
            **x,
            max_length=max_length, num_beams=num_beams,
            do_sample=do_sample, repetition_penalty=repetition_penalty,
            **kwargs
        )
    return sum_tokenizer.decode(out[0], skip_special_tokens=True)


text = 'test'
print(summarize(text, n_words=2))
