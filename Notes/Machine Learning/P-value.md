- The p-value is a statistical measure used in hypothesis testing to evaluate the strength of evidence against the null hypothesis.
- It quantifies the probability of observing a test statistic as extreme as, or more extreme than, the one observed, assuming the null hypothesis is true.
- The p-value represents the evidence against the null hypothesis. A smaller p-value indicates stronger evidence against the null hypothesis.
- The p-value ranges from 0 to 1, with values closer to 0 suggesting stronger evidence against the null hypothesis.
- Typically, a significance level (alpha) is chosen in advance, often set at 0.05. If the p-value is less than the significance level, we reject the null hypothesis in favor of the alternative hypothesis.
- Conversely, if the p-value is greater than the significance level, we fail to reject the null hypothesis, indicating insufficient evidence to support the alternative hypothesis.
- It is important to note that failing to reject the null hypothesis does not imply that the null hypothesis is true; it simply means there is not enough evidence to suggest otherwise.
- The p-value alone does not provide information about the magnitude or practical significance of the observed difference. It only assesses the statistical significance.
- Researchers must consider the context, effect size, and practical implications when interpreting the results of hypothesis testing.
- It is crucial to understand that statistical significance does not equate to practical significance. A statistically significant result may not necessarily be meaningful or important in the real world.
- The interpretation of p-values should always be done in conjunction with effect sizes, confidence intervals, and domain-specific knowledge.
- It is also worth noting that p-values can be influenced by sample size, effect size, variability, and the chosen statistical test.
- It is good practice to report the p-value alongside effect sizes and confidence intervals to provide a comprehensive understanding of the results.

### Calculate P-value 
the two sided is only used as the one side is dangerous as it takes the probability of the value from the current state of Null hypothesis to the zero which could be wrong. 

in the other side two sided take cares of extreme events. p-value = $$P(current event)+ P(another simialr rare event) + P(very extreme case)$$
we can conclude these into following 
1. The probability random chance would result in the observation 
2. the probability of observing something else that is equally rare 
3. the probability of observing something else rare or extreme. 

we add part 2  and part 3 as it seems rare to have the current state but it is not that rare when other states as just as rare as the current state. Remember that null hypothesis assume that state or the output is not due to the special treatment or special events rather than out of randomness and different events affected the current state or no relations between the different events. 


- you can calculate p-value from F using multiple formulas from T-test to ANOVA test. 
###  P-value of continues distribution of data
in order to calculate this we just get the area under the curve / total area under the curve. 
so when given a question of a specific value that belongs to the curve we can rewrite the same question in another format. 
`does this spacefic value is far away from the mean so we can reject the idea it came from the current distribution or there could be a better curve that fits this data more.`
- This involves considering the tail probabilities on both ends of the distribution of the test statistic.
### P-hacking

- P-hacking is a misuse of data analysis to find patterns in data that can be presented as statistically significant when in fact there is no real underlying effect.
- P-hacking can be unintentional or intentional, and it can lead to an increase in the number of false positives, misleading results, bias, and waste of resources.
- P-hacking can be avoided by pre-registering your study design and analysis plan before collecting data.
- P-hacking can also be avoided by using a holdout set of data to test your hypothesis.
- P-hacking can also be avoided by using Bayesian methods instead of frequentist methods.

instead of adding more observations to the data you have instead use the data in the power analysis to determine the correct sample size.


