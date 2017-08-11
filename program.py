from POSTagger import *
import time
import random


def read_data(path):
    """ str -> list
    loads data from a csv file
    removes redundant info
    :param path:
    :return: list of sentences
    """
    data = []
    with open(path, 'r') as f:
        sentence = []
        for line in f:
            a = line.split(' ')
            if len(a) < 3:  # end of sentence
                data.append(sentence)
                sentence.insert(0, ['', '##'])
                if sentence[-1][0] != '.':
                    sentence.append(['.', '.'])
                sentence = []
            else:
                sentence.append([a[0].strip(), a[1].strip()])
    return data


def concat_sentences(corpus):
    seq = ['']
    for sent in corpus:
        words = [p[0] for p in sent]
        seq.extend(words[1:-1])
    seq.append('.')
    return seq


def get_words(data):
    ret = []
    for s in data:
        ret.append([p[0] for p in s])
    return ret


def read_tags(path):
    t = []
    with open(path, 'r') as f:
        for line in f:
            tokens = line.strip().split(',')
            t.append(tokens[0])
    return t


def check_eol(data):
    dot_count = 0
    anomaly_count = 0
    for sentence in data:
        if sentence[-1][0] == '.':
            dot_count += 1
        else:
            anomaly_count += 1
    print("{0} out of {1} has '.' as their terminal state.".format(dot_count, len(data)))


def concat_chunks(record):
    """ list -> str
    convert words of a sentence to its values
    :param record: list of words and their pos tag
    :return: a sentence
    """
    words = [x[0] for x in record]
    return ' '.join(words)


def test_model(mod, data):
    total = 0  # total number of words
    correct = 0  # correctly tagged
    s = np.zeros(len(data))
    i = 0
    for sentence in data:
        ws = []
        ts = []
        for chunk in sentence:
            ws.append(chunk[0])
            ts.append(chunk[1])
        predicted_tags = mod.decode(ws)
        c = np.sum(np.array(ts[1:]) == np.array(predicted_tags[1:]))
        correct += c
        l = len(sentence) - 1
        total += l
        s[i] = c / l
        print("{0} : [length => {1}, corrects => {2}] | Accuracy => {3:.2%}".format(i, l, c, s[i]))
        i += 1
    word_acc = correct / total
    print("per word accuracy on test data is : {:.2%}".format(word_acc))
    sentence_acc = np.mean(s)
    print("average sentence accuracy on test data is : {:.2%}".format(sentence_acc))
    return word_acc, sentence_acc


def experiment_p1():
    train = read_data('train.txt')
    test = read_data('test.txt')
    dic = TextDictionary()
    dic.add(train)
    dic.add(test)

    model = HMM(dic.get_tags(), dic.get_words())
    tic = time.time()
    model.estimate(train)  # estimate the probabilities
    toc = time.time()
    print('estimation time: {0:.2f}'.format(toc - tic))
    model.save()
    test_model(model, test)


def experiment_p2():
    train = read_data('train.txt')
    test = read_data('test.txt')
    dic = TextDictionary()
    dic.add(train)
    dic.add(test)

    model = HMM(dic.get_tags(), dic.get_words())
    model.rand_init()
    random.shuffle(train)
    model.estimate(train[:-10])
    tic = time.time()
    model.train(concat_sentences(train[-10:]), max_iter=50)
    toc = time.time()
    print('learning time: {0:.2f}'.format(toc - tic))
    model.save()
    test_model(model, test)


if __name__ == '__main__':
    experiment_p2()

    print('hello, world!')
