IDEA 1:
Look at quotes which are given more media coverage (must choose in terms of numOccurrences or just quote counts). Try determining what makes a quote be popular:
- Speaker: occupation, genre, age, ???            ---------> May want to compare these stats to those of the whole population in the country of origin
- Text: consider quotes length, word count to see if some words are more common in viral quotes (may want to try with PCA or encodings to classify if quotes are provocative or not, and a general topic)
- Date: some times of year papers are more prone to publishing? Maybe relate to some real-world events (try extracting variance of date per quote too)
- Newspaper: do these stats change for different newspapers?


"Bias inference anaylsis" 
https://www.researchgate.net/profile/Ali-Minai/publication/267559458_Online_News_Media_Bias_Analysis_using_an_LDA-NLP_Approach/links/570b2cf808aea66081376d8b/Online-News-Media-Bias-Analysis-using-an-LDA-NLP-Approach.pdf
Aim may be to help politicians make their quotes more viral
May want to try make predictions about when quote will become viral (after 1h, 2h, ...)




IDEA 2:
Try to look at correlations between the stock market and quotes (in terms of date, argument discussed, ...)



Please use the following repository structure:

- Cache [dir]: Contains output files from long computations.
- Data [dir]: Contains the dataset as provided by the course. Includes quotebank's json files and the speaker attributes parquet file.
- Data Analysis [jupyter notebook]: The jupyter on which the analysis takes place. May be worth duplicating multiple times at the beginning such that each person has its own and we avoid git conflicts.

- Are we allowed to organize our code as to have external .py files? Because it sure will be a mess if we put all function definitions in the jupyter, but if we must, we will.

NOTE: we can sync Data and Cache folder using google drive. Normalement j'ai partagé avec vous un lien à mon espace de stockage epfl, sur votre mail epfl.
Par contre, si on se trompe et on efface accidentellement un fichier cache qui prend long à tourner, il n'est pas backed up. Du coup il est perdu et faudra le rerun.
