# site_classification_kaggle
Site classification task

The dataset contains 300 000 urs of websites and an associated category among 20 given labels. This may be for instance "Product", "Category", "Search", "Checkout", "Store locator" or the leftover category "Other". The goal is to carry out a multi-label classification task based on this dataset that is able to handle new urls for classification.

To evaluate our model, the chosen metrics is the accuracy score. We may either consider the global accuracy score, or the per-category accuracy score. Actually, the dataset is inbalanced, since some categories are much more represented than others. We can either make the hypothesis that it is non-biased, and faithful to the real-life distribution over website categories. Or we may think the dataset is biased towards certain categories. The per-category accuracy score could be more interesting for we might want to have a minimal SLA on every possible category. Moreover, a high accuracy score could only mean that we are efficient on an overrepresented category, and we might be weaker on the other ones. Thus we jointly consider the overall acccuracy score and the per-category accuracy score.

A first analysis of the dataset shows that urls can be split into three parts:
- The website or prefix representing the company that owns the website
- The path of the url on this website
- The "end of the url", containing other accurate pieces of information, called the action or c_vars

The approach was to vectorize each of those parts separately, and then apply a ML merger model. The simple multi-label logistic regression was adopted here but it is still subject to improvements and new tests.

The three parts were vectorized the same way. For each part (let's say the path), we split it into atomic chunks, and create a map M from a chunk to a score/probability of being part of an url of category CAT, for every category CAT. We displayed two different ways ("unbalanced" mode and "balanced" mode) of computing those scores, depending on if we wish to account for the priors on categories probabilities (due to the unbalanced nature of the dataset) or not. Once the map is created, we simply average the embedding of all chunks (i.e. the 20-dimensional arrays carrying the scores for every category) to get a 20-dimensional array.

Once all parts are vectorized, we stack them into a 60-dimensional vector and feed them to a model (here multi-class logistic regression). The model training is done with 5 stratified folds to maintain category distributions among th folds. Each fold splits the dataset into two parts: 80 % goes to train, the remaining 20% to test. Train is evenly split into two parts: one for estimating mappings M for each of prefix, path and action, the other one for model training. Then we estimate mappings on the whole train and carry out model predictions on test, yielding to a metrics score (overall and per-category). We finally average those metrics scores across the folds to get a final estimation of our model's performance.

We obtained the following results:

TODO

To obtain them again, one can only run all the cells of the enclosed Jupyter Notebook in the natural order. Results will pop within minutes.

Next improvements could include:

TODO
