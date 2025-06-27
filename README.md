# Views Predictor Project

This repo contains the code to analyze songs (better if they are from the same artists, they are more homogeneus) and train a model, a ResNet by default, to predict if they will surpass a thrashold of views or not (basically if they will become popular).

<br>

<!-- TABLE OF CONTENTS -->
  ### Table of contents
  <ol>
    <li>
      <a href="#my-test">My test</a>
      <ul>
        <li><a href="#results-on-validation-set">Results on Validation set</a></li>
        <li><a href="#results-on-test-set">Results on Test set</a></li>
      </ul>
    </li>
    <li>
      <a href="#contacts">Contacts</a>
    </li>
  </ol>

<br>

## My test
The project has been test with 63 songs of Bad Bunny taken by 5 of his most famous albums. I set the threshold at 300.000 views, using these configurations:
 <ol>
    <li>Sample Rate: 44.1 kHz</li>
    <li>Lenght of sample: 15sec</li>
    <li>Number of samples per song: 4</li>
    <li>Model: ResNet</li>
    <li>Number of epochs: 50</li>
  </ol>

<br>
<br>

### Results on Validation set
The results obtained in the Validation set:
<br>
Songs corrected classified:  36  on a total of  41
<br>
Accuracy:  0.8780487804878049
<br>
total loss:  1.2867190837860107  and loss per song:  0.031383392287463674
<br>
Tot predicted &lt;300M views: 24 on a total of &lt;300M songs: 27
<br>
Tot predicted >300M views:  17 on a total of >300M soongs:  14
<br>
The confusion matrix obtained is: 

![Confusion Matrix for Validation set](assets/valid_cm.png)

<br>
<br>


### Results on Test set
Then I tested the model in the Test set, made by 
![Confusion Matrix for Validation set](assets/valid_cm.png)



<br>
<br>

## Contacts
Matteo Zaramella - matteozara98@gmail.com

<br>
<br>
