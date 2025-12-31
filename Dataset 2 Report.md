Dataset 2 report – model comparison and learning curve

Dataset 2 investigates a balanced binary classification problem (200 class-0 and 200 class-1 samples) with eight numerical features. Missing values are imputed with the median and all features are standardised. The data is split into 300 training and 100 test samples, and four classifiers are compared: logistic regression, k-nearest neighbours (k = 5), random forest and a support vector machine (SVM).

Cross-validation on the training set shows that KNN has the lowest mean accuracy (≈0.93). Logistic regression performs better with a mean CV accuracy of about 0.97. SVM reaches ≈0.987 mean CV accuracy, while random forest achieves perfect cross-validation performance with a mean of 1.00 and zero standard deviation, indicating that it consistently classifies all folds correctly as shown in the bar graph chart below.


<img width="696" height="494" alt="image" src="https://github.com/user-attachments/assets/d2e939e0-bbe4-414d-aed5-adb9f7a9e5b5" />



Test-set performance mirrors these trends. KNN attains 0.97 accuracy and misclassifies three positive samples as class 0. Logistic regression achieves 0.99 accuracy, with only one positive misclassified. Both SVM and random forest reach 1.00 test accuracy and have confusion matrices with no errors, correctly identifying all 50 negatives and 50 positives. 

<img width="377" height="384" alt="image" src="https://github.com/user-attachments/assets/f830f112-cfb0-41ea-a273-8fcafb0363c3" />
<img width="377" height="384" alt="image" src="https://github.com/user-attachments/assets/7b51c1c3-9eba-4d1c-af13-df4e5c8fd661" />
<img width="377" height="384" alt="image" src="https://github.com/user-attachments/assets/13859419-5ee4-49b8-ad07-2c23aaa57247" />
<img width="377" height="384" alt="image" src="https://github.com/user-attachments/assets/9997b0a3-f5a5-404a-b826-3b695185230e" />

Although SVM and random forest tie on the test set, random forest is preferred overall because it also has perfect cross-validation accuracy.

A learning curve is generated for the random forest model, using training sizes from 5 up to 320 samples. The training accuracy is 1.00 for all sizes, showing that the random forest model can fit the training data perfectly well. The cross-valida+on curve starts around 0.63 at 5 training samples and rises quickly; once the training size exceeds 10 samples, the random forest consistently achieves more than 70% accuracy. The first +me when it exceeded 70% accuracy at n = 7, it dropped below 70% again at n = 9, and only above n = 10 do a consistent increase in accuracy takes place as indicated in the learning curve below.

With around 15–20 training samples the cross-validation accuracy already exceeds 0.90, and from about 30–40 samples onwards it stabilises above 0.98, approaching the near-perfect regime seen at larger training sizes.

<img width="1064" height="734" alt="image" src="https://github.com/user-attachments/assets/5e13b341-c022-4a85-99e2-02d8dceafcea" />


Overall, this analysis shows that all four models can learn the pattern in dataset 2 very effectively, but random forest offers the best combination of high and stable performance on both cross-validation and the held-out test set, so it is chosen as the best model to classify Dataset 2.



