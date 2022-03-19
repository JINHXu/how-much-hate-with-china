


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

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

df = pd.read_csv('data.csv')
df = df[df['tweet'].notna()]
X = df.tweet.values


input_ids, attention_masks, labels = prepare_dataset(
    X, np.zeros(len(X)), tokenizer, max_length=400)
dataset = TensorDataset(input_ids, attention_masks, labels)

y_pred = np.zeros(len(X))

flat_logits = run_bert_model(
    model, dataset, batch_size=8, extras=False)
print(flat_logits)
y_pred = np.argmax(flat_logits, axis=1).flatten()
print(y_pred)

df['cardiffnlp-RoBERTa-preds'] = y_pred
df.to_csv(args.output)

