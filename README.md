
# EM - Mixture Model (Topic Minig)

Utilize a Natural Language Processing (NLP) model with a mixture model approach to extract the main words (keywords) from a document. The Expectation-Maximization (EM) algorithm is applied to the background model to iteratively refine the probability distributions and identify the most significant terms.

$$
    \text{Assume}: \quad p(\Theta _d)=p(\Theta _B) = 0.5p(\Theta _d)=p(\Theta _B) = 0.5 
$$

$$
    \text{E-Step}: \quad p^{(n)}(z=0|w) = \frac{p(\Theta _d)p^{(n)}(w|\Theta _d)}{p(\Theta _d)p^{(n)}(w|\Theta d) + p(\Theta _B)p(w|\Theta B)} 
$$

$$
    \text{M-Step}: \quad p^{(n+1)}(w|\Theta_d) = \frac{c(w,d)p^{(n)}(z=0|w)}{\sum_{w' \in V} c(w',d)p^{(n)}(z=0|w')}
$$

## Usage/Examples
Set the ***override*** to ***True*** if you want to generate new output files.

```python
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

```

## Output
```
                 Word distribution    Background distribution
0         año:0.052809529703449876     el:0.10310729232250658
1       méxico:0.04841116861080652     de:0.07283063590237399
2         pri:0.046041601199958926   que:0.032183993185437576
3    nacional:0.046041601199958926    en:0.031980297026036074
4     político:0.04272980962054629     y:0.025591644753898003
5        mayo:0.041909827010015444    él:0.023480611829191513
6     mexicano:0.04128343357204001      a:0.02142513240250361
7      partido:0.04043215006168144   uno:0.018406725676826784
8    candidato:0.04043215006168144    del:0.01596237176400874
9        país:0.039340461914093794   ser:0.013221732528424873
10     mercado:0.03889466508181738   por:0.010499611125513869
11    gobierno:0.03844337935370845     su:0.00942557683048776
12     ciento:0.037054528553900457   con:0.009018184511684752
13  miércoles:0.036579160404708655  para:0.008888559682974705
14      millón:0.03609716899502946    no:0.008203399874078738
```