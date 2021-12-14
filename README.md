# site_classification_kaggle
Site classification task

**Task**

The dataset contains 300 000 urs of websites and an associated category among 20 given labels. This may be for instance "Product", "Category", "Search", "Checkout", "Store locator" or the leftover category "Other". The goal is to carry out a multi-label classification task based on this dataset that is able to handle new urls for classification.

**Metrics**

To evaluate our model, the chosen metrics is the accuracy score. We may either consider the global accuracy score, or the per-category accuracy score. Actually, the dataset is unbalanced, since some categories are much more represented than others. We can either make the hypothesis that it is non-biased, and faithful to the real-life distribution over website categories. Or we may think the dataset is biased towards certain categories. The per-category accuracy score could be more interesting for we might want to have a minimal SLA on every possible category. Moreover, a high accuracy score could only mean that we are efficient on an overrepresented category, and we might be weaker on the other ones. Thus we jointly consider the overall acccuracy score and the per-category accuracy scores.

**Ideas and model**

A first analysis of the dataset shows that urls can be split into three parts:<br>
- The website or prefix representing the company that owns the website<br>
- The path of the url on this website<br>
- The "end of the url", containing other accurate pieces of information, called the actions or c_vars<br>

The approach was to vectorize each of those parts separately, and then apply a ML merger model. The simple multi-label logistic regression was used here but it is still subject to improvements and new tests.

The three parts were vectorized the same way. For each part (let's say the path), we split it into atomic chunks, and create a map M from a chunk to a score/probability of being part of an url of category CAT, for every category CAT. We displayed two different ways ("unbalanced" mode and "balanced" mode) of computing those scores, depending on if we wish to account for the priors on category probabilities (due to the unbalanced nature of the dataset) or not. Once the map is created, we simply average the embedding of all chunks (i.e. the size-20 arrays carrying the scores for every category) to get a size-20 array.

Once all parts are vectorized, we stack them into a size-60 vector and feed those vectors to a ML model (here multi-class logistic regression). The model training is done with 5 stratified folds to maintain category distributions among the folds. Each fold splits the dataset into two parts: 80 % goes to train, the remaining 20% to test. Train is evenly split into two parts: one for estimating mappings M for each of prefix, path and action, the other one for model training. Then we estimate mappings on the whole train and carry out model predictions on test, yielding metrics scores (overall and per-category). We finally average those metrics scores across the folds to get a final estimation of our model's performance.

**Results**

We obtained the following results:

*Unbalanced mode*

Overall accuracy: 0.5164668989402215<br>
Accuracy by category:<br>
{'Appointments / booking': 0.45,<br>
 'Brand image': 0.7473601718282877,<br>
 'Careers & applications': 1.0,<br>
 'Cart': 0.7816389698458119,<br>
 'Category': 0.6553837342497136,<br>
 'Checkout': 0.5120805762944707,<br>
 'Confirmation': 0.971321501265004,<br>
 'Favorites / wishlist': 0.8551282051282051,<br>
 'Form': 0.6499999999999999,<br>
 'Formations / services': 0.9704724409448818,<br>
 'Help / support': 0.8502615694164989,<br>
 'Home': 0.6244410190674144,<br>
 'Information / legals': 0.5911504424778762,<br>
 'My account': 0.6454901960784314,<br>
 'Offers & services': 0.9195402298850575,<br>
 'Other': 0.46674467708885914,<br>
 'Press / news': 0.8248022598870056,<br>
 'Product': 0.18902031135594446,<br>
 'Search': 0.8308021448508278,<br>
 'Store locator': 0.7675956542276807}
 
*Balanced mode*

Overall accuracy: 0.5901507120350076<br>
Accuracy by category:<br>
{'Appointments / booking': 0.05,<br>
 'Brand image': 0.7321104112849389,<br>
 'Careers & applications': 1.0,<br>
 'Cart': 0.5852513877804771,<br>
 'Category': 0.4670103092783505,<br>
 'Checkout': 0.5226667426038872,<br>
 'Confirmation': 0.9546714793889934,<br>
 'Favorites / wishlist': 0.8705128205128206,<br>
 'Form': 0.5166666666666666,<br>
 'Formations / services': 0.9527559055118109,<br>
 'Help / support': 0.9180684104627768,<br>
 'Home': 0.6619413768019002,<br>
 'Information / legals': 0.7531904983698183,<br>
 'My account': 0.5980392156862745,<br>
 'Offers & services': 0.9933333333333334,<br>
 'Other': 0.5893520974402181,<br>
 'Press / news': 0.9157627118644067,<br>
 'Product': 0.4447710592638449,<br>
 'Search': 0.9408345987371582,<br>
 'Store locator': 0.8554558337269722}

To obtain them again, one can only run all the cells of the enclosed Jupyter Notebook in the natural order. Results will pop within minutes.

**Ideas of improvements**

Next improvements could include:

- Perform error analysis to improve features and focus on errors and categories with a weak accuracy score
- Clean maps from chunks to category scores: remove figures, explicit categories of words, stop words etc
- Create a system for merging similar words or words with the same grammatical root/meaning/translation Ex: search <-> searches, search <-> recherche, knitwear <-> sportswear etc. Is useful to handle new expressions in the url, namely actions. Use stemming or pre-trained embedding models like Word2Vec.
- Improve the path chunk probability computations by taking into account the prior/bias on the category
- Grid-search hyperparameters (Logistic l2 penalization, threshold on minimal occurrence for chunk relevance) etc
- Try scaling inputs before feeding to the penalized Logistic Regression
- Try other models: MLPClassifier, non-linear classifiers (Random Forest/Gradient Boosting etc) ...
- Dig in the specificity of a given url chunk for a given category -> use TF-IDF-like features. For instance, chunks like ".fr" or "us" are frequent in all the different categories, there are not discriminant. Find a way of implementing this in the vecto
- Combine our "balanced" and "unbalanced" modes to leverage the strengths of both of them. Merge them, or simply average the predict_proba of the two models
- Correct the randomness of fold splits to be able to get perfectly comparable results (the models can still remain random during training though)
