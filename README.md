
# EM - Mixture Model (Topic Minig)

NLP model to mine the main words of a document using a mixture model. The background model is used to perform the Expectation-Maximization algorithm to get the new probabilities of the distribution.

$$
E-Step = p^{(n)}(z=0|w) = \frac{p(\Theta _d)p^{(n)}(w|\Theta _d)}{p(\Theta _d)p^{(n)}(w|\Theta d) + p(\Theta _B)p(w|\Theta B)}
$$



