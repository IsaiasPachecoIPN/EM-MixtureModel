
# EM - Mixture Model (Topic Minig)

Utilize a Natural Language Processing (NLP) model with a mixture model approach to extract the main words (keywords) from a document. The Expectation-Maximization (EM) algorithm is applied to the background model to iteratively refine the probability distributions and identify the most significant terms.

$$
    \text{Assume} = \hspace{1cm} p(\Theta _d)=p(\Theta _B) = 0.5p(\Theta _d)=p(\Theta _B) = 0.5 \\

    \text{E-Step} = \hspace{1cm} p^{(n)}(z=0|w) = \frac{p(\Theta _d)p^{(n)}(w|\Theta _d)}{p(\Theta _d)p^{(n)}(w|\Theta d) + p(\Theta _B)p(w|\Theta B)} \\

    \text{M-Step} = \hspace{1cm} p^{(n+1)}(w|\Theta _d) = \frac{c(w,d)p^{(n)}(z=0|w)}{\sum_{w'\epsilon V}c(w',d)p^{(n)}(z=0|w')}
$$



