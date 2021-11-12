# Increasing the chances of your speech to go viral: an interrogation on what helps catch the attention of modern day fickle newsreaders 
**Authors:** Andrea Oliveri, C�lina Chkroun, Kaourintin Tamine, Mattia Lazzaroni

## Abstract 
In a world where the internet grants people access to audiences of millions of people, we want to identify what factors can be linked to someone reaching huge audiences. A metric of how many people are reached by a speaker is the number of individual quotations in newspapers. By analysing the distribution of extremely viral quotes we want to determine what characteristics are important and build guidelines, if possible, to optimize the chance of a quote reaching a wider range of people. These guidelines would constitute very interesting tools in politics and research for example to better choose the topic and the speaker delivering a concept to make them more likely to be quoted. 

## Research Questions:

#### Are there any identifiable characteristics of the speaker which make it more "quotable"?
Age of the speaker? Gender? Occupation? Are there characteristics that makes us be able to say this person's speech will be more quoted than this other one, and allow us to say that changing this parameter will make the person more likely to have impact.

#### Does the number of words in a quote or the topic influence its success?
We will also explore if the textual content of the quote has an impact on the virality. We will mainly investigate two factors: the number of words in the quote and its topic. 

## Datasets:
We have used two separate datasets:
- [`Quotebank`](https://zenodo.org/record/4277311#.YYpVGWDMJhE): corpus of quotations from news extracted using Quobert between 2015 and 2020.
- [`Wikidata`](https://www.wikidata.org/wiki/Wikidata:Main_Page): a database containing structured information about subjects in Wikimedia. In this project we will use it to gather information about the speaker of the quotes, such as gender, nationality and occupations.


## Methods

### 1. Collect data from Wikidata
The first basic step of this project is to extract all needed information from Wikidata.
To do so, the qids of all speakers in Quotebank are collected, as well as the ones for which the quote attribution was ambiguous.
With these we can filter out rows and columns we are not interested in from the provided [`speaker_attributes.parquet`](Data/speaker_attributes.parquet), and query from Wikidata the link counts of ambiguous speakers.
Finally, we can query Wikidata for human-readable English labels of the speaker attributes we are interested in.

### 2. Dealing with large dataset
Due to the large size and heavy compression of the Quotebank dataset, working with it presents two challenges:

1) Dataset can't be entirely fit in RAM
2) Parsing dataset requires a long time

To deal with the first issue, we decompress and load the lines from the json files one by one.

To deal with the second issue, a caching system to disk was implemented to store results of long computation. This system is implemented as an easy-to-use Python decorator (for a detailed explaination, read the doctring of `cache_to_file_pickle` in [src/utils.py](src/utils.py)).

Note that, after filtering, the speaker attributes data could easily be stored entirely in RAM while working on Quotebank line by line.

### 3. Exploring the data

#### Analysis number of occurences of quotes
We first analyzed the number of occurences of all the quotes present in Quotebank and defined the threshold above which a quote will be considered as viral.

#### Explore speaker's features
We explored the raw distribution of the speaker's features as well as the distributions obtained when weighting by the number of different quotes of the speaker and by the total number of occurrences of all his quotes. Further analysis looked at the co-occurences of different values for feature for which multiple values are given for a single speaker (for exemple the co-occurrence matrix of occupations assigned to a speaker), as well as the co-occurences between values of different features.

#### Length of quotes
We analyzed the distributions of lenghts of all quotes in Quotebank.

#### Extraction and visualization of quote topics
At first, we tried to extract the topics of the quote using `Latent semantic analysis`<sup>[1](https://en.wikipedia.org/wiki/Latent_semantic_analysis) [2](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html) [3](https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a)</sup>.
However, we found out by implementing this method that it was too computationally heavy and required too much RAM. 

Upon further research, we stumbled upon a (great) package called [`BERTopic`](https://github.com/MaartenGr/BERTopic)<sup>[1](https://towardsdatascience.com/dynamic-topic-modeling-with-bertopic-e5857e29f872)</sup>, implementing all the tools we needed to quickly extract the main topics present in the Quoteback dataset, visualize them and their importance, ...
Moreover, part of the computations can be hugely accelerated using a GPU. Using only 1% of our quotes, we successfully trained the model, extracted and visualized the main topics.

### 4. Training models
In order to try to predict the potential of a new quote to be viral, we wanted to both train classifiers for the labels viral/not-viral and a regressor for the number of occurrences.  

We though about using Logistic regression, Support vector machines or Linear Regression as classifiers for their easy interpretability.
For the regressor, we plan to use linear regression for the same reason.

To train the models, each line of Quotebank dataset will be converted into a feature vector (or discarded if feature is missing) and a binary label for classifiers or an interger output for the regressor. Training will be performed on a randomly sampled 70% subset of these lines (this also includes the validation set for hyperparameters tuning) and the rest will be used for testing.

We will then try to see if we can apply weight penalties to filter out the least relevant features.

## Proposed timeline
By 19/11/2021:

1. Explore further the possible correlations between features
2. Train BERTopic on larger portion of dataset
3. Try training BERTopic with parameter `compute_probability = True` such that we can predict not only most likely topic but probability of each topic of each quote

By 01/12/2021:

4. Create feature vectors and labels for quotes in the dataset
5. Train classifiers and regressor, optimize hyperparameters and analyse performance on test set
6. Use regularizarion to filter out least relevant features

By 17/12/2021:

7. Draw conclusions, if possible, from learned feature coefficients
8. Write data story

## Organization within the team
Within the team, the organization will be as follow (number refer to those in [Proposed Timeline](#proposed-timeline), note that people will work in parallel on different objectives):
- *Andrea:* 4, 5, 6, 7, 8
- *C�line:* 2, 3, 7, 8
- *Kaourintin:* 1, 4, 7, 8
- *Mattia:* 5, 6, 7, 8

## Repository structure

- [Cache](Cache): Contains results of long computations.
- [Data](Data): Contains the provided datasets (Quotebank's json files and [`speaker_attributes.parquet`](Data/speaker_attributes.parquet)).
- [src](src): Contains utilitary modules implementing:
    - Feature extraction: [feature_extraction.py](src/feature_extraction.py)
    - Visualization: [plot.py](src/plot.py)
    - Management of data: [utils.py](src/utils.py)
- [Project.ipynb](Project.ipynb): Notebook in which the analysis is performed.