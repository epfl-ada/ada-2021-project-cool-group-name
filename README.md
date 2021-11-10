
# Increasing the chances of your speech to go viral : an interrogation on what helps catch the attention of modern day fickle newsreaders 



## Abstract 
In a world where a simple tweet from someone can drastically change things for millions online, we want to identify what causes can be linked to someone having an impact on huge audiences. A metric of this is the number of individual quotations in newspapers around the world. By analysing the distribution of extremely viral quotes we want to determine what characteristics are important and build guidelinesn, if possible, to optimize the chance of a quote to reach a wider range of people. These guidelines would constitute very interesting tools for politicians and researchers specifically to help structure their speeches and works. (100/150 words for now).


## **Research Questions:**

#### **Are the extremely viral quotes predominantly random? Does high popularity stem mainly from random interest swings from the public?**
There are many trends that seem totally random in our society, from songs (Gnam Gnam style) to memes that overnight explode in popularity. Naturally, newspaper quotes are generally centered around more serious topics and might not be affected by these random phenomenons. It is quite an interesting question to try to quantify by "how much" these random trends are present in the quotation world.

#### **Are there any identifiable interesting characteristics from which we could deduce valuable lessons as to how to make oneself "quotable"?**
Keywords? Age of the speaker? Gender? Occupation? What are the characteristics that makes us be able to say this person's speech will be more quoted than this other one, and allow us to say that changing this parameter will make the person more likely to have impact.
Other characteristics will be explored like the date, correlation to real world events, nationality, newspaper type and the variance of date per quote (distribution of quote repetition).

## Methods

#### Initial data scraping (is scraping the right word here)
We have two separate datasets, the wikiData dataset which contains the information on persons/speakers (date of birth, gender, occupation...). And Quotebank, a corpus of quotations from a decade of news extracted using Quobert. https://zenodo.org/record/4277311#.YYpVGWDMJhE. The first basic step/method for this project is to fuse these two datasets into an exploitable structure for our data analysis. 

#### "Bias inference analysis" (implemented and dropped)
https://www.researchgate.net/profile/Ali-Minai/publication/267559458_Online_News_Media_Bias_Analysis_using_an_LDA-NLP_Approach/links/570b2cf808aea66081376d8b/Online-News-Media-Bias-Analysis-using-an-LDA-NLP-Approach.pdf
Aim may be to help politicians make their quotes more viral
May want to try make predictions about when quote will become viral (after 1h, 2h, ...)
We tried implementing this method but it was computationally too heavy especially memory wise (we have 16 gigas of ram and it was insufficient). Anyways, we prefered using
BerTopic which provided more useful functionalities

#### Using bertTopic (NLP based technique)
bertTopic allows us to classify the quotation set into topics. It makes a lot of sense to split the quotes into topic groups, for data visualization as well as for model training and other methods that might want the quotes to be split by topic. 

#### Logistic regression/random Forest/Linear Regression
We plan on parsing the processed data and convert each line into a feature vector and associate them with a viral/not viral label.
We will try to see if we can apply penalties to identify if some features are relevant or not.  

 
#### Proposed Timeline
Milestone 2 (Friday 12th november) : finish the fusing of the two datasets, finish the exploratory data analysis, implement bertTopic to split quotes and visualize and start thinking about the data story we want to start working on for the rest of the semester. 

#### Organization within the team
The second milestone on exploratory data analysis was very hard to split into even packets of work. Most functions had to be rethought and rewritten more than once and this meant that it was hard for someone to try things on their side until the data structure was fixed in place. 



Please use the following repository structure:

- [Cache](Cache) [dir]: Contains output files from long computations.
- [Data](Data) [dir]: Contains the dataset as provided by the course. Includes quotebank's json files and the speaker attributes parquet file.
- [Project](Project.ipynb) [jupyter notebook]: The jupyter on which the analysis takes place. May be worth duplicating multiple times at the beginning such that each person has its own and we avoid git conflicts.

- Are we allowed to organize our code as to have external .py files? Because it sure will be a mess if we put all function definitions in the jupyter, but if we must, we will.

NOTE: we can sync Data and Cache folder using google drive. Normalement j'ai partagé avec vous un lien à mon espace de stockage epfl, sur votre mail epfl.
Par contre, si on se trompe et on efface accidentellement un fichier cache qui prend long à tourner, il n'est pas backed up. Du coup il est perdu et faudra le rerun.
