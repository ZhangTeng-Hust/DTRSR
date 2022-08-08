# DTRSR
A deep transfer regression method based on seed replacement considering balanced domain adaptation

### Contribution
- 1. Marginal distribution difference and conditional distribution difference are measured simultaneously by balanced distribution adaptation loss, which achieves more sufficient learning of domain knowledge.
- 2. Seed replacement is applied to integrate the knowledge of source domain and target domain into the new data set in the form of cluster structure and cluster center, which achieves more active learning of domain knowledge.
- 3. The DTRSR proposed in this paper provides an adequate and active learning framework for regression tasks in terms of model, data, and loss function.

### Structure
The flow chart of the DTRSR is shown as follows. In terms of structure, there are four parts, including (a) Structure freezing and parameters transfer, (b) Deep feature extraction, (c) Seed replacement and (d) Fusion Loss Function. 
<div align=center>
<img src=https://github.com/ZhangTeng-Hust/DTRSR/blob/main/IMG/DTRSR.png>
</div>

### Result
The average MAEs of all methods in 18 tasks are compared as follows.
<div align=center>
<img src=https://github.com/ZhangTeng-Hust/DTRSR/blob/main/IMG/LeiDa.png>
</div>
From the radar maps shown above, it can be seen that the red lines are in the inner circle and are all at a greater distance from the outer circle, which indicates that DTRSR is effective.

## Special Reminder
No reproduction without permission！！！

This work is currently being submitted to Elsevier, so no individual or organization can fork this repository until it has been reviewed.

## Added instructions
The robot machining errors data set is a private data set constructed by our team, which is not convenient to be disclosed. We hope you can understand.
The results of the proposed method are not guaranteed to be the best, but are sufficient to prove the effectiveness of the method.
After the manuscript is accepted, the data that support the findings of this study are available from the corresponding author upon reasonable request.
Links to accepted article will also be added here.

## Updata
A Share Link – a personalized URL providing 50 days' free access to the article have been created. Anyone clicking on this link before September 27, 2022 will be taken directly to the final version of this article on ScienceDirect, which you are welcome to read or download. No sign up, registration or fees are required.

The personalized Share Link: https://authors.elsevier.com/c/1fYM-3OWJ90-Mu
