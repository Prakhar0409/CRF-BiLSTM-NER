import optparse
import torch
import pickle
from torch.autograd import Variable
import sys
from utils import *
from model import BiLSTM_CRF
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

MODELS_PATH = "snapshots/"
OUTFILE = 'predict.txt'
VOCAB_FILE = 'models/mapping.pkl'
TRAIN_FILE = '../dataset/train.txt'
TEST_FILE = '../dataset/test.txt'
DATA_FILE = '../dataset/ner.txt'

optparser = optparse.OptionParser()
optparser.add_option("-w", "--word_dim", default="300",type='int', help="Word embedding dimension")
optparser.add_option("-W", "--word_lstm_dim", default="200",type='int', help="LSTM hidden layer size")
optparser.add_option("-p", "--pre_embeddings", default=None,help="pretrained embeddings file path")
optparser.add_option("-r", "--reload", default="0",type='int', help="pretrained model path to reload")
optparser.add_option('--name', default='test',help='model name')
opts = optparser.parse_args()[0]

parameters = {}
parameters['lower'] = True
parameters['zeros'] = False
parameters['word_dim'] = opts.word_dim
parameters['word_lstm_dim'] = opts.word_lstm_dim
parameters['pre_embeddings'] = opts.pre_embeddings
parameters['reload'] = opts.reload == 1
parameters['name'] = opts.name

use_gpu = False
if torch.cuda.is_available():
    use_gpu = True

name = parameters['name']
model_name = MODELS_PATH + name

if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)

lower = parameters['lower']
zeros = parameters['zeros']

## List of sentences where each sentence is a list of [word, tag]
## [  [[word1, tag1], [word2,tag2]]  
##      [[word1, tag1], [word2,tag2]]
##   ]

# sentences = load_sentences(DATA_FILE, lower, zeros)
# def foo():
train_sentences = load_sentences(TRAIN_FILE, lower, zeros)
test_sentences = load_sentences(TEST_FILE, lower, zeros)


dico_words, word_to_id, id_to_word = word_mapping(train_sentences, lower)
dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)

train_data = prepare_dataset(train_sentences, word_to_id, char_to_id, tag_to_id, lower)
test_data = prepare_dataset(test_sentences, word_to_id, char_to_id, tag_to_id, lower)

print("%i - %i sentences in train - test." % (len(train_data), len(test_data)))

### Loading pretrained embeddings if available
all_word_embeds = {}
if opts.pre_embeddings:
    for i, line in enumerate(codecs.open(opts.pre_embeddings, 'r', 'utf-8')):
        s = line.strip().split()
        if len(s) == parameters['word_dim'] + 1:
            all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])

word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_to_id), opts.word_dim))

for w in word_to_id:
    if w in all_word_embeds:
        word_embeds[word_to_id[w]] = all_word_embeds[w]
    elif w.lower() in all_word_embeds:
        word_embeds[word_to_id[w]] = all_word_embeds[w.lower()]

print('Loaded %i pretrained embeddings.' % len(all_word_embeds))

with open(VOCAB_FILE, 'wb') as f:
    mappings = {
        'word_to_id': word_to_id,
        'tag_to_id': tag_to_id,
        'char_to_id': char_to_id,
        'parameters': parameters,
        'word_embeds': word_embeds
    }
    pickle.dump(mappings, f)

print('word_to_id: ', len(word_to_id))
model = BiLSTM_CRF(vocab_size=len(word_to_id), tag_to_ix=tag_to_id, embedding_dim=parameters['word_dim'], 
                hidden_dim=parameters['word_lstm_dim'], use_gpu=use_gpu, char_to_ix=char_to_id, pre_word_embeds=word_embeds)
                   
if parameters['reload']:
    model.load_state_dict(torch.load(model_name))
if use_gpu:
    model.cuda()

learning_rate = 0.015
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
loss = 0.0
best_test_F = -1.0
count = 0

sys.stdout.flush()

def save_model(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model, save_path)

def evaluate(model, datas, best_F):
    prediction = []
    save = False
    new_F = 0.0
    for data in datas:
        ground_truth_id = data['tags']
        str_words = data['str_words']
        list_list_chars = data['chars']
        
        word_lengths = [len(c) for c in list_list_chars]
        # print(list_list_chars, word_lengths)
        max_word_length = max(word_lengths)
        padded_words = np.zeros((len(word_lengths), max_word_length), dtype='int')
        for i, c in enumerate(list_list_chars):
            padded_words[i, :word_lengths[i]] = c
        padded_words = Variable(torch.LongTensor(padded_words))

        words = Variable(torch.LongTensor(data['words']))
        if use_gpu:
            val, out = model(words.cuda(), padded_words.cuda(), word_lengths)
        else:
            val, out = model(words, padded_words, word_lengths)
        predicted_id = out
        sentence = []
        for (word, true_id, pred_id) in zip(str_words, ground_truth_id, predicted_id):
            sentence.append([word, id_to_tag[true_id], id_to_tag[pred_id]])
        prediction.append(sentence)

    predf = OUTFILE
    with open(predf, 'w') as f:
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
    save = False
    new_F = totalF1
    if new_F > best_F:
        best_F = new_F
        save = True
    return best_F, new_F, save


model.train(True)
for epoch in range(1, 100):
    print('Epoch Number: ', epoch)
    for i, index in enumerate(np.random.permutation(len(train_data))):
        count += 1
        data = train_data[index]    # pick a sentence
        model.zero_grad()

        sentence_in = data['words'] # [w1,w2,w3]
        sentence_in = Variable(torch.LongTensor(sentence_in))
        tags = data['tags']         # [x, y ,z]
        list_list_chars = data['chars']      # [[c1,c2,c3], [c2,c4]]

        word_lengths = [len(c) for c in list_list_chars]

        max_word_length = max(word_lengths)
        padded_words = np.zeros((len(word_lengths), max_word_length), dtype='int')
        for i, c in enumerate(list_list_chars):
            padded_words[i, :word_lengths[i]] = c
        padded_words = Variable(torch.LongTensor(padded_words))

        targets = torch.LongTensor(tags)
        
        if use_gpu:
            neg_log_likelihood = model.neg_log_likelihood(sentence_in.cuda(), targets.cuda(), padded_words.cuda(), word_lengths)
        else:
            neg_log_likelihood = model.neg_log_likelihood(sentence_in, targets, padded_words, word_lengths)
        loss += neg_log_likelihood.data[0] / len(data['words'])
        neg_log_likelihood.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
        optimizer.step()
        
    # Evaluate
    model.train(False)
    print('count: ',count)
    best_test_F, new_test_F, save = evaluate(model, test_data, best_test_F)
    if save:
        # torch.save(model, model_name)
        save_model(model, MODELS_PATH, 'best', epoch)
    
    sys.stdout.flush()
    model.train(True)

    adjust_learning_rate(optimizer, lr=learning_rate/(1+0.05*count/len(train_data)))
