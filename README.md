# Pain-recognition, First Steps	
Neural network to recognise pain and pain levels.	
Python programs written by Paul Van Ghent, then 'tweaked' a little by myself.	

## Introduction
The original aim of this project was to look at how well a neural network can recognise human emotions from a series of images and from there to investigate the implications and applications from the research. 

The research was then focused on pain identification providing a potentially usable system in healthcare to help with identifying pain rather than relying on human recognition, opening up to human error.  The research is explained more indepth in the literature review in appendix a.  Several datasets were analysed to suggest the viability of studying pain by itself.

After downloading the Cohn-Kanade dataset, a previously unfound report was downloaded with the dataset which highlighted a problem area for emotion recognition (Kanade, 2000).  The report stated; using facial landmarks to identify emotions was not an accurate way to identify an emotion, because it only took in to consideration the movement of the face, and more is needed to identify specific emotion, including body movement, heart rate and other biological signals.  As such a system to monitor and identify an emotion  cannot be based purely on facial signals (Kanade, 2000).  

This brought an important point to my mind, because there are not only neural networks to identify an emotion, but also Support Vector Machines which use algorithms to classify an image from its general features (Gent, 2016) and also from landmarks, depending on what kernel is used, rather than just facial landmarks (or Action Units) to be taken and fed into it like a neural network would (such as a Feed Forward neural network).  Which would be better?  Would a newer kind of neural network be better, such as a convoluted neural network (CNN), which is better at handling visual data, be better suited to the task?  Before a system can be developed to identify a single emotion such as pain, the algorithms used must be looked at first, and then how to link biological signals from the subject to the facial movements to identify the correct emotion and its level can be decided on.



