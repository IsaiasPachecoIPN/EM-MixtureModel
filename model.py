import  pickle
import  utils
import  math
import  numpy       as np
import  pandas      as pd
from    bs4         import BeautifulSoup


class TopicMining:

    def __init__(self):
        """
        Constructor
        @param data: dataset
        @param vocabulary: vocabulary
        @param word_count_dataset: dataset with the word count
        @param word_count_model: model to predict the score
        """

        self.data                       = None
        self.preproced_data             = None
        self.vocabulary                 = None
        self.word_count                 = None
        self.word_count_probabilities   = None

    def load_dataset(self, path, verbose=False):

        """
        Load the dataset from a csv file
        @param path: path to the csv file
        @param verbose: print additional information
        """

        self.data = ""
        for doc in path:
            self.data += open(doc, 'r', encoding='utf-8').read()
        # self.data = [open(doc, 'r', encoding="UTF-8").read() doc for document in path]

    def parse_text(self, verbose=False):

        """
        Parse the thex fron the htm using BeautifulSoup
        """

        soup = BeautifulSoup(self.data, 'lxml')
        self.data = soup.get_text()    

        if verbose:
            print(f'Parsed text: {self.data}')

    def preprocess_text(self, remove_stopwords=True, remove_numbers=True, remove_punctuation=True,lemmatize_text=True,lower_text=True, stop_words_path=None, verbose=False, override=False):

        """
        Function to preprocess the text
        @param remove_stopwords: remove stopwords
        @param remove_numbers: remove numbers
        @param remove_punctuation: remove punctuation
        @param stop_words_path: path to the file with the stopwords
        """

        #Check if the text is already preprocessed
        try:
            if override:
                raise Exception("Override")
            with open('./output/preprocessed_text.txt', 'r') as f:
                self.data = f.read()
            print(f'Preprocessed text loaded')
        except:

            if lower_text:
                print(f'Preprocessing text: lower_text')
                self.data = utils.lower_text(self.data)

            if remove_stopwords:
                print(f'Preprocessing text: remove_stopwords')
                self.data = utils.remove_stopwords(self.data, stop_words_path)

            if remove_punctuation:
                print(f'Preprocessing text: remove_punctuation')
                self.data = utils.remove_punctuation(self.data)

            if remove_numbers:
                print(f'Preprocessing text: remove_numbers')
                self.data = utils.remove_numerical_values(self.data)

            if lemmatize_text:
                print(f'Preprocessing text: lemmatize_text')
                self.data = utils.lemmatize_text(self.data)

            #Save the preprocessed text
            with open('./output/preprocessed_text.txt', 'w') as f:
                f.write(self.data)

            print(f'Preprocessed text created')

        if verbose:
            print(f'Preprocessed text: {self.data}')



    def build_vocabulary(self, verbose=False, override=False):

        """
        Build the vocabulary from the dataset. If the vocabulary already exists, it will be loaded.
        @param verbose: print additional information
        @param override: override the existing vocabulary
        """

        #Check if vocabulary already exists
        try:
            if override:
                    raise Exception("Override")
            with open('./output/vocabulary.pkl', 'rb') as f:
                self.vocabulary = pickle.load(f)

            print(f'Vocabulary loaded')

        except:
            self.vocabulary = set()
            for text in self.data.split():
                self.vocabulary.add(text)

            #save vocabulary
            with open('./output/vocabulary.pkl', 'wb') as f:
                pickle.dump(self.vocabulary, f)

            print(f'Vocabulary created')

        if verbose:
            # print(f'Vocabulary: {self.vocabulary}')
            print(f'Vocabulary size: {len(self.vocabulary)}')

    def build_background_language_model_probabilities(self, verbose=False):

        """
        Build the language model for topic mining
        """

        print(f'Building language model for topic mining')

        #Word count
        self.word_count = {}
        for word in self.data.split():
            if word in self.word_count:
                self.word_count[word] += 1
            else:
                self.word_count[word] = 1

        # print(f'Word count: {self.word_count}')

        self.word_count_probabilities = {}
        #Total number of words
        N = len(self.data.split())

        #Normalize Each word count
        for word, freq in self.word_count.items():
            self.word_count_probabilities[word] = freq/N

        sum = 0
        for word, freq in self.word_count_probabilities.items():
            sum += freq
        
        print(f'Sum: {sum}')

        #Order the word count
        self.word_count_probabilities = dict(sorted(self.word_count_probabilities.items(), key=lambda item: item[1], reverse=True))

        #Print the first n words
        count = 0
        count_breaker = 5
        for word, freq in self.word_count_probabilities.items():
            print(f'{word}: {freq}')
            count += 1
            if count == count_breaker:
                break    
    
    def calculate_em_steps(self, steps=1, verbose=False):

        """
        Calculate the EM steps using the background language model
        """

        #Probability of the topic language model
        p_theta_d = 0.5 
        
        #Probability of the background language model
        p_theta_B = 0.5

        w_p_theta_d = {word: 0.25 for word in self.vocabulary}
        
        global_log_likelihood = 0

        for step in range(steps):
            p_w_z_0_steps = {}

            for word in self.vocabulary:
                #Calculate the E step
                p_w_z_0 = ( p_theta_d * w_p_theta_d[word] ) / ( p_theta_d * w_p_theta_d[word] + p_theta_B * self.word_count_probabilities[word] )
                p_w_z_0_steps[word] = p_w_z_0

            #Sum of the p_w_z_0 * wcounts
            wc_p_w_z_0_sum = sum(self.word_count[word] * p_w_z_0_steps[word] for word in self.vocabulary)

            #Update the w_p_theta_d probabilities
            for word in self.vocabulary:
                # w_p_theta_d[word] = ( self.word_count[word] * p_w_z_0_steps[word] ) / wc_p_w_z_0_sum
                w_p_theta_d[word] = 1 - p_w_z_0_steps[word]

            # Calculate the log-likelihood
            log_likelihood = sum(self.word_count[word] * np.log(p_theta_d * w_p_theta_d[word] + p_theta_B * self.word_count_probabilities[word]) for word in self.vocabulary)

            if global_log_likelihood == log_likelihood:
                break
            else:
                global_log_likelihood = log_likelihood

            if verbose:
                print(f'EM Step [{step}] Log-Likelihood: {log_likelihood}')

        #Sort the w_p_theta_d probabilities
        w_p_theta_d = dict(sorted(w_p_theta_d.items(), key=lambda item: item[1], reverse=True))

        #Remove the stop words
        stop_words = open('./src/spanish.txt', 'r', encoding='utf-8').read().splitlines()
        w_p_theta_d = {word: freq for word, freq in w_p_theta_d.items() if word not in stop_words}

        if verbose:
            print(f'Word distribution after {steps} EM steps:')

            #Print the first n words
            count = 0
            count_breaker = 15

            #Save the word distribution and the background distribution in a dataframe to compare
            df = pd.DataFrame()
            df['Word distribution'] = [f'{word}:{freq}' for word,freq in list(w_p_theta_d.items())[:count_breaker]]
            df['Background distribution'] = [f'{word}:{freq}' for word,freq in list(self.word_count_probabilities.items())[:count_breaker]]

            print(df)

            # for wd, bd in zip(w_p_theta_d.items(), self.word_count_probabilities.items()):
            #     print(f'{wd}: {bd}')
            #     count += 1
            #     if count == count_breaker:
            #         break

            # print(f'Background distribution')

            # count = 0
            # for word, freq in self.word_count_probabilities.items():
            #     print(f'{word}: {freq}')
            #     count += 1
            #     if count == count_breaker:
            #         break