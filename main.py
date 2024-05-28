from model import *

dataset_path = ['./src/e990519_mod.htm', './src/e990520_mod.htm', './src/e990521_mod.htm', './src/e990522_mod.htm']

override = False

nlp = TopicMining()
nlp.load_dataset(dataset_path)
nlp.parse_text()
nlp.preprocess_text(remove_stopwords=False, remove_numbers=True, remove_punctuation=True,lemmatize_text=True, lower_text=True, stop_words_path='./src/spanish.txt', verbose=False, override=override)
nlp.build_vocabulary(override=override, verbose=True)
nlp.build_background_language_model_probabilities(verbose=True)
nlp.calculate_em_steps(steps=500, verbose=True)
