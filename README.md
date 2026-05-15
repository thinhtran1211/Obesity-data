# Obesity-data
### Introduction

Obesity is a chronic condition characterized by a high level of body fat that can pose substantial risks to someone’s mental and physical health. Our data set estimates the obesity levels based on eating habits and physical condition. Obesity is a dynamic situation with many unique cases. For the purposes of this paper we have classified the condition into “High Risk” and “Manageable” levels. We will use the data set called “Estimation of Obesity Levels Based on Eating Habits and Physical Condition” obtained through the UC Irvine Machine Learning Repository. This data set contains variables related to health, socio-economic status, and lifestyle choices of individuals from Mexico, Peru, and Columbia, and their obesity levels.

Weight classifications are usually based on BMI, short for Body Mass Index, a general calculation based on a man or woman’s height and weight that attempts to show how much body fat they have. BMI levels within the range of 25-29.9 are considered manageable while BMI levels over 30 are classified as Obese. While there may not be an obvious semantic difference between “overweight” and “obese”, studies have shown that there is up to 91% higher risk of hospitalization and up to 70% risk of mortality compared to a manageable weight. The data set does not go in depth on how they classify the weight ranges, instead opting for a categorical variable that we will use as our response variable in the data set. The variable is `NObeyesdad`, which classifies an individual into weight classes. We will define "manageable weight" as a weight that falls into `Insufficient_Weight`, `Normal_Weight`, `Overweight_Level_I`, and `Overweight_Level_II`. "High-risk" weight will be represented as a weight that falls into `Obesity Type I`, `Obesity Type II`, and `Obesity Type III`.

**Research Question:** Can lifestyle choices such as eating habits, transportation, technology use, and smoking accurately predict a person’s weight level between “manageable” and “high-risk”?

**Description of Variables:**\
*Gender* - Categorical\
*Age* - Continuous\
*Height* - Continuous\
*Weight* - Continuous\
*Family_history_with_overweight* - Binary, Has a family member suffered or suffers from overweight?\
*FAVC* - Binary, Do you eat high caloric food frequently?\
*FCVC* - Integer, Do you usually eat vegetables in your meals?\
*NCP* - Continuous, How many main meals do you have daily?\
*CAEC* - Categorical, Do you eat any food between meals?\
*SMOKE* - Binary, Do you smoke?\
*CH2O* - Continuous, How much water do you drink daily?\
*SCC* - Binary, Do you monitor the calories you eat daily?\
*FAF* - Continuous, How often do you have physical activity?\
*TUE* - Integer, How much time do you use technological devices such as cell phone, videogames, television, computer and others?\
*CALC* - Categorical, How often do you drink alcohol?\
*MTRANS* - Categorical, Which transportation do you usually use?\
*NObeyesdad* - Categorical, Obesity Level categorized into Insufficient_Weight, Normal_Weight, Overweight_Level_I, Overweight_Level_II, Obesity_Type_I, Obesity_Type_II, Obesity_Type_III

### Methods and Results

**Logistic Regression (Shaheer Abbasi and Long Pham)**\
Our goal is to be able to predict, as accurately as possible, whether an individual is at risk based on certain predictors. A logistic regression model is appropriate for this response variable because it models the probability of a binary outcome. Using a multivariate approach allows us to have a different view of the data from all possible angles, giving us a more encompassing and accurate result. However, using a multivariate logistic regression will not explain certain variables well if they do not have linear relationships.

A basic multivariate logistic regression formula would be:\
$$p(X) = \frac{exp(\beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_3 + \dots + \beta_{n} X_{n})}{1 + exp(\beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_3 + \dots + \beta_{n} X_{n})}$$

We randomly split the data into 80% training and 20% testing and repeated this process 10 times to record the mean test error rate.

```{r}
obesity = read.csv("ObesityDataSet_raw_and_data_sinthetic 2.csv")

# Define our binary response categories
# High Risk = 1
# Manageable = 0
obesity$RiskLevel = ifelse(obesity$NObeyesdad %in% c("Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"), 1, 0)

# Convert categorical predictors to factors
cols = c("Gender", "FAVC", "CAEC", "SMOKE", "CALC", "MTRANS")
obesity[cols] <- lapply(obesity[cols], as.factor)

# Drop rare levels in CALC to prevent unseen level errors
obesity = obesity[obesity$CALC != "Always", ]
obesity$CALC = droplevels(obesity$CALC)

# Fit the model to obtain coefficients
obesity.glm = glm(RiskLevel ~ FAVC + NCP + CAEC + SMOKE + FAF + TUE + CALC + MTRANS, data = obesity, family = "binomial")

summary(obesity.glm)

# Train and test the data using K-fold cross validation
set.seed(1)
lg.error.rate = rep(0, 10)

for (i in 1:10){
  sample = sample.int(n = nrow(obesity),
  size = round(.80 * nrow(obesity)), replace = FALSE)
  train = obesity[sample, ]
  test = obesity[-sample, ]

  obesity.glm2 = glm(RiskLevel ~ FAVC + NCP + CAEC + SMOKE + FAF + TUE + CALC + MTRANS,data = train, family = "binomial")

  glm.pred = predict.glm(obesity.glm2, newdata = test, type = "response")
  yHat = glm.pred > 0.5
  lg.cm = table(test$RiskLevel, yHat)
  lg.error.rate[i] = (lg.cm[1,2] + lg.cm[2,1]) / sum(lg.cm)
}
```

The resulting model from our logistic regression:

$$p(X) = \frac{exp(-8.7355 - 0.0065(\text{GenderMale}) + 0.1204(\text{Age}) + 2.3593(\text{FAVCyes}) + 0.2097(\text{NCP}) - 1.8892(\text{CAECFrequently}) - 1.6723(\text{CAECno}) + 1.6245(\text{CAECSometimes}) + 0.4598(\text{SMOKEyes}) - 0.1927(\text{FAF}) - 0.2174(\text{TUE}) + 0.6152(\text{CALCno}) + 0.7674(\text{CALCSometimes}) - 0.5820(\text{MTRANSBike}) + 1.1059(\text{MTRANSMotorbike}) + 1.5946(\text{MTRANSPublic}) - 0.7168(\text{MTRANSWalking}))}{1 + exp(-8.7355 - 0.0065(\text{GenderMale}) + 0.1204(\text{Age}) + 2.3593(\text{FAVCyes}) + 0.2097(\text{NCP}) - 1.8892(\text{CAECFrequently}) - 1.6723(\text{CAECno}) + 1.6245(\text{CAECSometimes}) + 0.4598(\text{SMOKEyes}) - 0.1927(\text{FAF}) - 0.2174(\text{TUE}) + 0.6152(\text{CALCno}) + 0.7674(\text{CALCSometimes}) - 0.5820(\text{MTRANSBike}) + 1.1059(\text{MTRANSMotorbike}) + 1.5946(\text{MTRANSPublic}) - 0.7168(\text{MTRANSWalking}))}$$

Since we are trying to predict a categorical variable, we will use the confusion matrix formula:

$$Error = \frac{FalsePositives + FalseNegatives}{Total}$$

```{r}
lg.error.rate
mean(lg.error.rate)
```

After 10 iterations of the training and testing, the mean test error rate was 30.81%. This implies that a person's lifestyle choices such as FAVC, NCP, CAEC, smoking status, physical activity frequency, technology use, alcohol consumption, and transportation method can predict whether an individual is at "high-risk" or "manageable" obesity levels with roughly 69.19% accuracy.

**Linear Discriminant Analysis (Jordan Pho, Jack Tran, Isaac Kapeel)**\

LDA assumes that X is drawn from a multivariate normal distribution, where each class has its own mean μk, and both classes share the same spread (common covariance matrix Σ). LDA is very effective in binary classification, though due to the assumption of normality, it makes it so that predictors with categorical values should make it less effective. For example, predictors like MTRANS should make the model perform a little worse due to MTRANS being five categorical values. LDA has capabilities of dimension reduction, though we do not leverage that capability in this project.


Below is a typical LDA Multivariant Classifier:

$$\delta_k(x) = x^T\Sigma^{-1}\mu_k - \frac{1}{2}\mu_k^T\Sigma^{-1}\mu_k + \log(\pi_k)$$
We will first preprocess the data.
```{r}
library(MASS)

obesity = read.csv("ObesityDataSet_raw_and_data_sinthetic 2.csv")


obesity$RiskLevel = ifelse(obesity$NObeyesdad %in% c("Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"), "High_Risk", "Manageable")

cols = c("FAVC", "CAEC", "SMOKE", "CALC", "MTRANS")
obesity[cols] <- lapply(obesity[cols], as.factor)

# Drop rare levels in CALC to prevent unseen level errors
obesity = obesity[obesity$CALC != "Always", ]
obesity$CALC = droplevels(obesity$CALC)


```
We will randomly split training and test data. Then train the model and test it. This will be repeated 10 different times
```{r}
set.seed(1)
lda.error.rate = rep(0, 10)

for (i in 1:10){
  index = sample(1:nrow(obesity), size = 0.8 * nrow(obesity))
  train = obesity[index, ]
  test = obesity[-index, ]
  
  lda.fit = lda(RiskLevel ~ FAVC + NCP + CAEC + SMOKE + FAF + TUE + CALC + MTRANS,data = train)
  
  lda.pred = predict(lda.fit, test)$class
  lda.error.rate[i] = mean(lda.pred != test$RiskLevel)
}


```
We will then average the error rates
```{r}
lda.error.rate
mean(lda.error.rate)
```
After 10 iterations of the training and testing, the mean test error rate was 31.81%, having a slightly higher error rate in comparison logistic regression model. This means that through LDA, the model can accurately predict roughly 68.19% of the time.

Below is the confusion matrix and summary of the 10th run for the lda model

```{r}
table(lda.pred, test$RiskLevel)
lda.fit$scaling
```
