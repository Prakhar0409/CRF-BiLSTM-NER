import torch
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import pickle
from utils import *
import sys

TESTFILE = sys.argv[1]   #"../dataset/test.txt"
MODELFILE = sys.argv[2]
VOCAB_FILE = sys.argv[3]
PREDICTION_FILE = sys.argv[4]

with open(VOCAB_FILE, 'rb') as f:
    mappings = pickle.load(f)

word_to_id = mappings['word_to_id']
tag_to_id = mappings['tag_to_id']
id_to_tag = {k[1]: k[0] for k in tag_to_id.items()}
char_to_id = mappings['char_to_id']
word_embeds = mappings['word_embeds']
parameters = mappings['parameters']

use_gpu = False
if torch.cuda.is_available():
    use_gpu = True

lower = parameters['lower']
zeros = parameters['zeros']

test_sentences = load_sentences(TESTFILE, lower, zeros)
test_data = prepare_dataset(test_sentences, word_to_id, char_to_id, tag_to_id, lower)

model = torch.load(MODELFILE)

if use_gpu:
    model.cuda()

model.eval()
def eval(model, datas):
    prediction = []
    for data in datas:
        ground_truth_id = data['tags']      # there is no truth in this ground

        str_words = data['str_words']
        list_list_chars = data['chars']

        word_lengths = [len(c) for c in list_list_chars]
        max_word_length = max(word_lengths)
        padded_words = np.zeros((len(word_lengths), max_word_length), dtype='int')
        for i, c in enumerate(list_list_chars):
            padded_words[i, :word_lengths[i]] = c
        padded_words = Variable(torch.LongTensor(padded_words))

        dwords = Variable(torch.LongTensor(data['words']))
        if use_gpu:
            val, out = model(dwords.cuda(), padded_words.cuda(), word_lengths)
        else:
            val, out = model(dwords, padded_words, word_lengths)
        predicted_id = out
        sentence = []
        for (word, true_id, pred_id) in zip(str_words, ground_truth_id, predicted_id):
            sentence.append([word, id_to_tag[true_id], id_to_tag[pred_id]])
        prediction.append(sentence)

    with open(PREDICTION_FILE, 'w') as f:
        for sentence in prediction:
            for (word, true_id, pred_id) in sentence:
                f.write(word + ' ' + true_id + ' ' + pred_id + '\n')
            f.write('\n')

    pred1 = []
    labels = []
    print(tag_to_id)
    for sentence in prediction:
        for (word, true_id, pred_id) in sentence:
            pred1.append(pred_id)
            labels.append(true_id)

    conf_mat = confusion_matrix(labels, pred1)
    totalaccu = accuracy_score(labels, pred1)
    totalF1 = f1_score(labels, pred1, average='macro')
    print(conf_mat)
    print('totalaccu', totalaccu)
    print('totalF1', totalF1)

# print(test_data[0])
eval(model, test_data)
