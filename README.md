
# Increasing the chances of your speech to go viral : an interrogation on what helps catch the attention of modern day fickle newsreaders 



## Abstract 
In a world where a simple tweet from someone can drastically change things for millions online, we want to identify what causes can be linked to someone having an impact on huge audiences. A metric of this is the number of individual quotations in newspapers around the world. By analysing the distribution of extremely viral quotes we want to determine what characteristics are important and build guidelinesn, if possible, to optimize the chance of a quote to reach a wider range of people. These guidelines would constitute very interesting tools for politicians and researchers specifically to help structure their speeches and works. (100/150 words for now).


## **Research Questions: A list of research questions you would like to address during the project.**

#### **Are the extremely viral quotes predominantly random? Does high popularity stem mainly from random interest swings from the public?**
There are many trends that seem totally random in our society, from songs (Gnam Gnam style) to memes that overnight explode in popularity. Naturally, newspaper quotes are generally centered around more serious topics and might not be affected by these random phenomenons. It is quite an interesting question to try to quantify by "how much" these random trends are present in the quotation world.

#### **Are there any identifiable interesting characteristics from which we could deduce valuable lessons as to how to make oneself "quotable"?**
Keywords? Age of the speaker? Gender? Occupation? What are the characteristics that makes us be able to say this person's speech will be more quoted than this other one, and allow us to say that changing this parameter will make the person more likely impact.

IDEA 1:
Look at quotes which are given more media coverage (must choose in terms of numOccurrences or just quote counts). Try determining what makes a quote be popular:
- Speaker: occupation, genre, age, ???            ---------> May want to compare these stats to those of the whole population in the country of origin
- Text: consider quotes length, word count to see if some words are more common in viral quotes (may want to try with PCA or encodings to classify if quotes are provocative or not, and a general topic)
- Date: some times of year papers are more prone to publishing? Maybe relate to some real-world events (try extracting variance of date per quote too)
- Newspaper: do these stats change for different newspapers?






#### Methods

##### Initial data scraping (is scraping the right word here)
We have two separate datasets, the wikiData dataset which contains the information on persons/speakers (date of birth, gender, occupation...). And Quotebank, a corpus of quotations from a decade of news extracted using Quobert. https://zenodo.org/record/4277311#.YYpVGWDMJhE. The first basic step/method for this project is to fuse these two datasets into an exploitable structure for our data analysis. 

##### "Bias inference analysis" (nope ca on utilise plus)
https://www.researchgate.net/profile/Ali-Minai/publication/267559458_Online_News_Media_Bias_Analysis_using_an_LDA-NLP_Approach/links/570b2cf808aea66081376d8b/Online-News-Media-Bias-Analysis-using-an-LDA-NLP-Approach.pdf
Aim may be to help politicians make their quotes more viral
May want to try make predictions about when quote will become viral (after 1h, 2h, ...)

####Using Bert 
#### Proposed Timeline

#### Organization within the team




Please use the following repository structure:

- Cache [dir]: Contains output files from long computations.
- Data [dir]: Contains the dataset as provided by the course. Includes quotebank's json files and the speaker attributes parquet file.
- Data Analysis [jupyter notebook]: The jupyter on which the analysis takes place. May be worth duplicating multiple times at the beginning such that each person has its own and we avoid git conflicts.

- Are we allowed to organize our code as to have external .py files? Because it sure will be a mess if we put all function definitions in the jupyter, but if we must, we will.

NOTE: we can sync Data and Cache folder using google drive. Normalement j'ai partagé avec vous un lien à mon espace de stockage epfl, sur votre mail epfl.
Par contre, si on se trompe et on efface accidentellement un fichier cache qui prend long à tourner, il n'est pas backed up. Du coup il est perdu et faudra le rerun.
