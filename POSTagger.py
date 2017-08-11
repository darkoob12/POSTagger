import numpy as np
import sys


class HMM:
    def __init__(self, tags, words):
        self.tags = tags
        self.tags.insert(0, "##")  # initial state
        self.tags.append('.')  # terminal state
        self.words = words
        self.b = np.zeros((self.num_tags(), self.num_words()))
        self.a = np.zeros((self.num_tags(), self.num_tags()))

    def num_words(self):
        try:
            return len(self.words)
        except NameError:
            return 0

    def num_tags(self):
        try:
            return len(self.tags)
        except NameError:
            return 0

    def save(self):
        np.savetxt('hmm_a.csv', self.a, delimiter=',')
        np.savetxt('hmm_b.csv', self.b, delimiter=',')

    def load(self):
        self.a = np.loadtxt('hmm_a.csv', delimiter=',')
        self.b = np.loadtxt('hmm_b.csv', delimiter=',')

    def decode(self, seq):
        """ list -> list
        given a sequence of observed words this function predicts the sequence of tags corresponding to these
        words using Viterbi algorithm
        :param seq: words in a sentence
        :return: tags of the given words
        """
        assert type(seq) is list
        seq_len = len(seq)
        alpha = np.zeros((self.num_tags(), seq_len))

        def alpha_fcn():
            if j == 0 and t == 0:
                return 1
            elif j != 0 and t == 0:
                return 0
            else:
                return np.sum(alpha[:, t - 1] * self.a[:, j]) * self.b[j, self.words.index(seq[t])]

        path = []
        for t in range(seq_len):
            for j in range(self.num_tags()):
                alpha[j, t] = alpha_fcn()
            jp = np.argmax(alpha[:, t])
            path.append(self.tags[jp])

        return path

    def learn(self, data, set_random=False):
        if set_random:
            self.rand_init()  # random values for a and b
        for seq in data:
            self.train(seq)

    def train(self, seq, threshold=0.02, max_iter=5):
        """ list -> none
        This would learn the hmm parameters using forward-backward algorithm Given a set sentences with no tags
        :param threshold: threshold for change in parameters as termination criterion
        :param max_iter: maximum number of iterations
        :param seq: a sequence of sentences
        :return: None
        """
        # initialize parameters and variables
        alpha = np.ones((self.num_tags(), len(seq))) / self.num_tags()
        beta = np.ones((self.num_tags(), len(seq))) / self.num_tags()
        gamma = np.ones((self.num_tags(), self.num_tags(), len(seq))) / self.num_tags()

        def forward():
            for t in range(len(seq)):
                for j in range(self.num_tags()):
                    if t == 0 and j == 0:
                        alpha[j, t] = 1
                    elif t == 0 and j != 0:
                        alpha[j, t] = 0
                    else:
                        alpha[j, t] = np.sum(alpha[:, t - 1] * self.a[:, j]) * self.b[j, self.words.index(seq[t])]
                # normalize each column
                alpha[:, t] /= np.sum(alpha[:, t])

        def backward():
            T = len(seq) - 1
            for t in range(T - 1, -1, -1):
                for j in range(self.num_tags()):
                    if t == T and j == self.num_tags() - 1:
                        beta[j, t] = 1
                    elif t == T and j != self.num_tags():
                        beta[j, t] = 0
                    else:
                        beta[j, t] = np.sum(beta[:, t + 1] * self.a[:, j]) * self.b[j, self.words.index(seq[t + 1])]
                # normalize each column
                beta[:, t] /= np.sum(beta[:, t])

        def compute_gamma():
            for t in range(len(seq) - 1):
                sum_ = 0
                for i in range(self.num_tags()):
                    for j in range(self.num_tags()):
                        gamma[i, j, t] = alpha[i, t] * self.a[i, j] * self.b[j, self.words.index(seq[t])] * beta[
                            j, t + 1]
                        sum_ += gamma[i, j, t]
                gamma[:, :, t] /= sum_  # normalize for each time step

        iteration = 0
        while True:
            try:
                # E-Step decode using current estimates of a and b
                print('fwd\t', end='')
                sys.stdout.flush()
                forward()
                print('bwd\t', end='')
                sys.stdout.flush()
                backward()
                print('gamma\t', end='')
                sys.stdout.flush()
                compute_gamma()
                # M-Step re-estimate a and b using obtained hidden values
                a_hat = np.zeros(self.a.shape)
                b_hat = np.zeros(self.b.shape)

                # estimate a
                print('re-estimate')
                for t in range(len(seq)):
                    a_hat += gamma[:, :, t]
                denum = np.sum(a_hat, 1)
                for i in range(self.num_tags()):
                    a_hat[i, :] /= np.sum(a_hat[i, :])
                # estimate b
                for t in range(len(seq)):
                    k = self.words.index(seq[t])
                    b_hat[:, k] += np.sum(gamma[:, :, t], 1)
                for j in range(self.num_tags()):
                    b_hat[j, :] /= denum[j]

                b_hat += 0.5 / self.num_words()  # smoothing

                delta = np.max(np.max(a_hat - self.a), np.max(b_hat - self.b))
                self.a = a_hat
                self.b = b_hat
                print("iteration = {0} , delta = {1}".format(iteration, delta))
                iteration += 1
                if iteration > max_iter or delta < threshold:
                    break
            except ZeroDivisionError:
                print('wow')
                break
            except FloatingPointError:
                print('NO :(')
                break

    def create_alpha(self, seq):
        """ list -> (function,function)
        returns a recursive version of Forward algorithm which can compute the alpha value for any give state
        at any given time step
        :param seq: a sequence of observations
        :return:  a function that computes alpha and another that returns the current computed values of alpha
        """
        alpha = np.zeros((self.num_tags(), len(seq)))
        alpha[:] = np.NaN

        def fcn(j, t):
            if not np.isnan(alpha[j, t]):
                return alpha[j, t]
            if t == 0 and j == 0:
                alpha[j, t] = 1
            elif t == 0 and j != 0:
                alpha[j, t] = 0
            else:
                tmp = np.array([fcn(i, t - 1) for i in range(self.num_tags())])
                alpha[j, t] = np.sum(tmp * self.a[:, j]) * self.b[j, self.words.index(seq[t])]
            return alpha[j, t]

        def get_alpha():
            return alpha

        def get_likelihood():
            if np.isnan(alpha).any():
                fcn(-1, -1)
            return np.sum(alpha[:, -1])

        return fcn, get_alpha(), get_likelihood

    def rand_init(self):
        # random numbers
        self.b = np.random.rand(self.num_tags(), self.num_words())
        self.a = np.random.rand(self.num_tags(), self.num_tags())
        # normalize probabilities
        self.a[:, 0] = 0  # initial state
        self.a[-1, :] = 0  # terminal state
        for j in range(self.num_tags()):
            if j != 46:
                self.a[j, :] /= np.sum(self.a[j, :])
            self.b[j, :] /= np.sum(self.b[j, :])

    def estimate(self, data):
        """
        using a corpus with grand truth tags estimates the state transition probabilities and
        the observation emission probability for each state using MLE
        :param data: a training list of sentences with grand truth POS tags
        """

        def append_word(pair):
            t = pair[1]
            if t in states:
                states[t].append(pair[0])
            else:
                states[t] = [pair[0]]

        states = dict()
        # counting number of transitions
        for sentence in data:
            for i in range(len(sentence) - 1):
                append_word(sentence[i + 1])  # count emissions
                for j in range(i, len(sentence)):
                    t1 = sentence[i][1]
                    t2 = sentence[j][1]
                    if t1 not in self.tags or t2 not in self.tags:
                        continue
                    self.a[self.tags.index(t1), self.tags.index(t2)] += 1

        # normalization and smoothing
        for i in range(1, self.num_tags() - 1):
            tag = self.tags[i]
            if tag not in states:
                self.b[i, :] = 1 / self.num_words()
                continue
            total = len(states[tag])
            wc = {self.words.index(x): states[tag].count(x) for x in set(states[tag])}
            for w in range(self.num_words()):
                ec = wc[w] + 1 if w in wc else 1
                tc = total + self.num_words() if w in wc else self.num_words()
                if w in wc:
                    self.b[i, w] = ec / tc

        for i in range(np.size(self.a, 0)):
            self.a[i, :] = (self.a[i, :] + 1) / (np.sum(self.a[i, :]) + self.num_tags())


class TextDictionary:
    def __init__(self):
        self.words = dict()
        self.tags = dict()

    def add(self, corpus):
        try:
            if self.words is None:
                self.words = dict()
            if self.tags is None:
                self.tags = dict()
        except NameError:
            self.words = dict()
            self.tags = dict()

        for sentence in corpus:
            for pair in sentence:
                word = pair[0]
                tag = pair[1]
                if word in self.words:
                    self.words[word] += 1
                else:
                    self.words[word] = 1
                if tag in self.tags:
                    self.tags[tag] += 1
                else:
                    self.tags[tag] = 1

    def __len__(self):
        return len(self.words)

    def get_words(self):
        return list(self.words.keys())

    def get_tags(self):
        return list(self.tags.keys())
