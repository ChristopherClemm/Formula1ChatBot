# Formula1ChatBot
What we aimed to make:
For our final project, we elected to construct a chat bot to answer questions regarding Formula 1.  We started off bold and wanted it to handle lots of different questions about drivers, races, and the constructors but we eventually got swapped with work and decided to focus on more of the drivers and races.

Tech Specs:
Bot was written in Python
Tensorflow, nltk, sklearn, csv, and regular expression packages
Data from Kaggle
Naive bayes, Feature extraction and engineering to sort and organize datasets into bags of words
Regex to parse through our datasets to find relevant information

Sources:
We used lots of different resources to help us create this bot. Primarily used knowledge from our HW2 exploration. We also used a youtube tutorial that helped us set up a basic bot which allowed us to apply neural networks. 
Link: https://www.techwithtim.net/tutorials/ai-chatbot/part-1/

Goal:
Provide a reasonably comprehensive chat bot which provides up to date information relevant to the user query.

How we did it:
To start our bot, we created a neural network that would take in data from a json file we created that had tagged some questions with things we would like our bot to respond to. We stemmed the words and used the bag of words technique to create our data that we would pass into the neural network to train. This part was largely helped by online tutorials and really help us

Accessing the data through our Kaggle database was a challenge because all of the data was spread across multiple csv’s. 

We used tags called intents that we would use to assign to potential questions the user would ask. For example the “nthplace” tag could be triggered through phrases like "Who got 3nd place in the in " and "Who got  at the “. This was trained by our neural net that would take the input and try to link it to tags. 

Once a tag has been identified, then we would search for information that the tag would need. For some it was just trying to find a name, but for others we would need to find position, year, name, and race track/grand prix. 

To search for this information we created more models that were naive bayes based and trained it with some of the different data sets. Some issues would come up  because the datasets were large and sometimes lead to bad predictions because there were few features that could be based on. We wanted to handle lots of cases where someone could say

Regular expressions came to the rescue and allowed us to easily extract place and year from our input. We would then use the year to narrow down the projected races based on the year and feed all of the races and circuits into a new model. We did some feature engineering to add different attributes of circuits to our TfidfVectorizer which made the look up a lot more accurate than before. 

Biggest problems and challenges:  
In order to create a functional chatbot, we need to first be able to understand the intentions behind each user input and we don’t have lots of actual user data to train it on.  
Extracting relevant information from a query
Machine learning in general is hard but luckily there are lots of great resources/tools
Parsing accurate data from csv files(We were stuck for an error because of the way a csv files was formatted)
Lots of issues with no data or gaps in data which made it difficult to spot some errors

What can we improve on next time?
More modularization of code (didn’t have a clear vision at the beginning which could have helped if we planned a little more)
We would include more error checking because sometimes are code would like to yell at us
Better data, dataset was good but was spread over lots of different csv files which made looking things up quite tedious and annoying, could have commented code better and used more functions but yeah
Commenting code would have saved us lots of time and we should have used more function calls to help abstract code. 


Takeaways from the project and overall experience
	This was our first time truly using lots of different machine learning applications to the test in a project environment. We learned lots and were able to apply lots of the different techniques such as stemming,naive bayes, bag of words which was satisfying to see. Overall the experience was good and we are proud of what we were able to accomplish. 
