# Increasing the chances of your speech to go viral : an interrogation on what helps catch the attention of modern day fickle newsreaders 
Authors: Andrea Oliveri, Célina Chkroun, Kaourintin Tamine, Mattia Lazzaroni


## Abstract 
In a world where a simple tweet from someone can drastically change things for millions online, we want to identify what causes can be linked to someone having an impact on huge audiences. A metric of this is the number of individual quotations in newspapers. By analysing the distribution of extremely viral quotes we want to determine what characteristics are important and build guidelines, if possible, to optimize the chance of a quote reaching a wider range of people. These guidelines would constitute very interesting tools in politics and research for example to better choose the content of of speeches to make them more likely to be quoted. 


## **Research Questions:**

#### **Are the extremely viral quotes predominantly random? Does high popularity stem mainly from random interest swings from the public?**
There are many trends that seem totally random in our society, from songs  to memes that overnight explode in popularity. Naturally, newspaper quotes are generally centered around more serious topics and might not be affected by these random phenomenons. It is quite an interesting question to try to quantify by "how much" these random trends are present in the quotation world.

#### **Are there any identifiable interesting characteristics from which we could deduce valuable lessons as to how to make oneself "quotable"?**
Keywords? Age of the speaker? Gender? Occupation? What are the characteristics that makes us be able to say this person's speech will be more quoted than this other one, and allow us to say that changing this parameter will make the person more likely to have impact.
Other characteristics will be explored like the date, correlation to real world events, nationality, newspaper type and the variance of date per quote (distribution of quote repetition).

#### **Does the number of words in a quote influence its success? And are there any topics more recurrent in more viral quotes?**
We will also explore if the textual content of the quote has an impact on the virality. At first, we chose to look at the word count of the quote and, later on, we wanted to extract the topic(s) of the quotes to see if some of them are more recurrent in the viral quotes or not. It would be interesting to find if the viral quotes have a particular length, because it can be an useful information for example for a politician that wants to attract people's attention. Same thing for the topics, it could be informative for a candidate at presidency to see which topics are given more attention to people.

## Methods

### 1. Collect data of interest from multiple datasets
We have two separate datasets:
- [`Wikidata`](https://www.wikidata.org/wiki/Wikidata:Main_Page): dataset which contains the information on persons/speakers (date of birth, gender, occupation...).
- [`Quotebank`](https://zenodo.org/record/4277311#.YYpVGWDMJhE): corpus of quotations from a decade of news extracted using Quobert.

The first basic step/method for this project is to fuse these two datasets into an exploitable structure for our data analysis.

### 2. Dealing with large dataset
**TO DO ANDREAAAAAAAAAA**

### 3. Exploring the data

#### Analysis of number of occurences of the quotes
We first analyzed the number of occurences of all the quotes present in the dataset in order to define the threshold above which a quote will be considered as viral.

#### Explore speaker's features
We also conducted an analysis on speaker's features. 
At first, we explored the raw distribution of the speaker's features and, after that, we weighted the distribution in 2 different ways (by the number of quotes and by the total number of occurences of quotes per speaker). Further analysis were performed to look at the co-occurences of different values per feature when multiple values are given per speaker. The co-occurences between different features were also analyzed.

#### Length of Quotes
We also analyzed the length of quotes by plotting the distribution of length of quotes and its proportion.

#### Extraction and Visualization of the Quote Topics
At first, we tried to extract the topics of the quote using a [Bias inference analysis](https://www.researchgate.net/profile/Ali-Minai/publication/267559458_Online_News_Media_Bias_Analysis_using_an_LDA-NLP_Approach/links/570b2cf808aea66081376d8b/Online-News-Media-Bias-Analysis-using-an-LDA-NLP-Approach.pdf) approach to find quote topics.
Nevertherless, we found out by implementing this method that it was computationally too heavy especially memory-wise.

We then decided to try to extract the topics using [BERTopic](https://towardsdatascience.com/dynamic-topic-modeling-with-bertopic-e5857e29f872) .
BERTopic is a NLP based technique allowing us to classify the quotation set into topics and on top, many useful functionalities are provided with it.
Indeed, it makes a lot of sense to split the quotes into topic groups, for data visualization as well as for model training and other methods that might want the quotes to be split by topic. 

### 4. Training a classifier
In order to try to predict the potential of a new quote to be viral, we wanted to train a classifier. We though about using Logistic regression, Random Forest or Linear Regression as a model.
To train it, we plan on parsing the processed data and convert each line into a feature vector and associate them with a viral/not viral label.
We will then try to see if we can apply penalties to identify if some features are relevant or not.
 
## Proposed Timeline
Next steps to go on with the project:
- Explore further correlation between features
- Train bert on larger portion of dataset
- Try training bert with compute_probability parameter = True such that we can predict not only most likely topic but probability of each topic of each quote
- Select meaningful features
- Merge these data into a feature vector and associate a label viral/not viral to it
- Train multiple types of classifier to see if features meaningful and if classifier works well on a test set
- Use regularizarion on well-performing models to reduce number of non-zero coefficients and filter actually useful features
- Try to conclude something from the found results and write the data story

## Organization within the team
Within the team, the organization will be as follow:
- Member 1: Continue the data exploration (feature's correlation, retraining Bert).
- Member 2: Training multiple classifier, select correct method and hyperparameters.
- Member 3: Focus on visualization of the project's results and think about how presenting them into the final data story.
- Member 4: Writing of the data story.

## Repository organization

- [Cache](Cache) [dir]: Contains output files from long computations.
- [Data](Data) [dir]: Contains the provided dataset (quotebank's json files and speaker attributes parquet file).
- [Project](Project.ipynb) [jupyter notebook]: The jupyter in which the analysis takes place.
- [Source](src) [dir] Contains .py modules where utilitary functions are implemented.
    - For feature extraction: feature_extraction.py
    - For visualization: plot.py
    - To manage the data: utils.py

## Questions

**TO ADD IF ANY QUESTIONS, COULD BE USEFUUUUUUUUUUUUUUUL**
