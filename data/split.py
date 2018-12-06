f = open('ner.txt', 'r', encoding = "ISO-8859-1")

# print(f.read())

con = f.readlines()
# print(con)
data = []
s = []
t = []
for l in con:
    print(l)
    l = l[:-1]
    ws = l.split()
    print(ws)
    if len(ws) != 2:
        data.append((s,t))
        s = []
        t = []
        continue
    s.append(ws[0])
    t.append(ws[1])

training_ratio = 0.9
training_len = int(training_ratio * len(data))
training_data, testing_data = data[:training_len], data[training_len:]
print(training_data)

train_f = open('train.txt', 'w')
for (sent,tags) in training_data:
    for word,tag in zip(sent, tags):
        train_f.write(word + ' ' + tag+'\n')
    train_f.write('\n')

train_f.close()

test_f = open('test.txt', 'w')
for (sent,tags) in testing_data:
    for word,tag in zip(sent, tags):
        test_f.write(word + ' ' + tag+'\n')
    test_f.write('\n')

test_f.close()
