# Project Apollo - ReadME


<!-- Badges Table -->
<table style="background-color: #f7f7f7;">
  <tr>
<td><a href="https://shareholder.ford.com/Investors/Home/default.aspx"><img 
src="https://upload.wikimedia.org/wikipedia/commons/d/d8/Ford_logo.svg" width="80" height="60"></a><td>
<td><a href="https://investors.att.com/"><img 
src="https://upload.wikimedia.org/wikipedia/commons/3/31/AT%26T_logo_2016.svg" width="80" height="60"></a><td>
<td><a href="https://investor.starbucks.com/ir-home/default.aspx"><img src="https://upload.wikimedia.org/wikipedia/commons/d/d6/Starbucks_logo.jpg" width="80" height="40"></a><td>
  </tr>
</table>



# Tools Used

<!-- Badges Table -->
<table style="background-color: #f7f7f7;">
  <tr>
    <td><a href="https://pandas.pydata.org/"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Pandas_logo.svg/1200px-Pandas_logo.svg.png" width="80" height="40"></a></td>
    <td><a href="https://numpy.org/"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/NumPy_logo.svg/1280px-NumPy_logo.svg.png" width="80" height="40"></a></td>
    <td><a href="https://scipy.org/"><img 
src="https://upload.wikimedia.org/wikipedia/commons/b/b2/SCIPY_2.svg" width="80" height="40"></a></td>
    <td><a href="https://matplotlib.org/"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/84/Matplotlib_icon.svg/1200px-Matplotlib_icon.svg.png" width="80" height="40"></a></td>
    <td><a href="https://seaborn.pydata.org/"><img src="https://seaborn.pydata.org/_static/logo-wide-lightbg.svg" width="80" height="40"></a></td>
    <td><a href="https://scikit-learn.org/stable/"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/1024px-Scikit_learn_logo_small.svg.png" width="80" height="40"></a></td>
  </tr>
</table>


<!-- #region -->


<a name="top"></a>
## Table of Contents
- [Description](#Description)
- [Goal](#Goal)
- [Initial_Thoughts](#Initial_Thoughts)
- [Planning](#Planning)
- [Acquire](#Acquire)
- [Data_Dictionary](#Data_Dictionary)
- [Prepare](#Prepare)
- [Exploration](#Exploration)
- [Modeling](#Modeling)
- [Nural_Network](#Nural_Network)
- [Key_Findings](#Key_Findings)
- [Steps_to_Reproduce](#Steps_to_Reproduce)
- [Summary](#Summary)
- [Conclusion](#Conclusion)
- [Next_Steps](#Next_Steps)

<details>
<summary><strong>Description</strong></summary>

## Description

This project predicts the next quarter's revenue for U.S blue chip companies (Ford Motor Company, Starbucks, and ATT), based on 80 quarters (20 years worth) of various economic, social, political, and environmental factors.
    
<a href="#top">Return to Table of Contents</a>
</details>

<details>
    
<summary><strong>Goal</strong></summary>

## Goal

The goal is to use generic data available to the public rather than industry-specific data or company-specific data to make a good total revenue prediction, better than predicting the average.
    
<a href="#top">Return to Table of Contents</a>
</details>

<details>
<summary><strong>Initial_Thoughts</strong></summary>

## Initial_Thoughts

"Everything, Everywhere, all at once" — this profound quote encapsulates the concept of the interconnectedness of all things across different locations and scenarios, occurring simultaneously. With this in mind, our project focuses on uncovering influential features within data segments from diverse sectors, which can have substantial impacts on revenue performance. By acquiring these subtle insights, decision-makers can be equipped with advanced knowledge of the external factors influencing their revenues, enabling them to adapt their business strategies accordingly.

With the above in mind, some initial questions we had:
- Are our target variables normally distributed?
- What features are statistically significant to our targets?
- Can the same features work for multiple targets? (Targets tested separately)
- What are the impacts of negative and positive correlating features?

<a href="#top">Return to Table of Contents</a>
</details>

<details>
<summary><strong>Planning</strong></summary>

## Planning

Generated a range of innovative ideas for revenue-affecting features through productive brainstorming sessions. Organized tasks using a Kanban board, efficiently tracking their progress under categories like 'Needs to be done', 'Doing', and 'Done'. Collaboratively compiled and maintained a shared knowledge document, ensuring seamless dissemination of new information, ideas, and functions across the team. Set clear milestone due dates and benchmarks, providing a solid foundation for measuring progress and achieving project goals.

<a href="#top">Return to Table of Contents</a>
</details>

<details>
<summary><strong>Acquire</strong></summary>

## Acquire

During the "Acquire" phase of our project, we collected a rich dataset comprising 40 features meticulously sourced from over 17 distinct websites. Notable among them are:
- Federal Reserve Economic Data (FRED)
- Bureau of Labor & Statistics (BLS)
- Organization for Economic Cooperation and Development (OECD)
    
Bringing all this valuable data together, we created a unified and coherent dataframe. This comprehensive dataframe incorporates data spanning two decades, encompassing 80 quarters. Each row represents one quarter, containing all pertinent revenue figures and associated features. See data dictionary below:

<a href="#top">Return to Table of Contents</a>
</details>

<details>
<summary><strong>Data_Dictionary</strong></summary>

## Data_Dictionary

| Feature | Definition (measurement)|
|:--------|:-----------|
|adjusted_revenue| The total quarterly revenue of the target Company in dollars adjusted for inflation|
|Year| The year of the data |
|Quarter| The quarter of the data |
|Population|The U.S. population for the quarter| 
|Median_house_income| The median household income per quarter in dollars|
|Federal_fund_rate| The interest rate that U.S. banks pay one another top borrow or loan money overnight (percentage)|
|Unemp_rate| The unemployment rate, the number of residents without a job and looking for work divided by the total number of residents|
|Home_ownership_rate| Home ownership rate by population|
|Government_spending| Government spending in billions of dollars|
|Gdp_deflated| measures changes in the prices of goods and services produced in the United States|
|Cpi_all_items_avg| Measures price change experienced by urban consumers; the average change in price over time of a market basket of consumer goods and services|
|Avg_temperature| The avg temperature in fahrenheit for the quarter|
|Avg_precipitation| The avg rainfall in inches for the quarter|
|Palmer_drought_index| The magnitude of PDSI indicates the severity of the departure from normal soil moisture conditions|
|eci|The Employment Cost Index, is a quarterly measure of the change in the price of labor, defined as compensation per employee hour worked|
|dow|Quarterly Dow Jones Industrial average|
|P_election| If it is a presidential election year (1=yes)|
|Midterm_election| If it is a midterm election year (1=yes)|
|Violent_crime_rate|Violent crimes (involve force or threat of force) per 100,000 |
|Consumer_confidence_index| An indication of future developments based on households' responses 100+ being a positive outlook|
|Case_shiller_index| benchmark of average single-family home prices in the U.S., calculated monthly based on changes in home prices over the prior three months |
|Prime| The prime rate is the interest rate that commercial banks charge creditworthy customers|
|Man_new_order|Motor Vehicles and Parts, Monthly, Seasonally Adjusted (in millions of dollars)|
|Construction_res|Total amount spent on residential construction (in millions of dollars)|
|CLI|The composite leading indicator, designed to provide early signals of turning points in business cycles|
|Soy|Soy bean prices|
|Misery_index|The measure of economic distress felt by everyday people, due to the risk of (or actual) joblessness combined with an increasing cost of living| 

<a href="#top">Return to Table of Contents</a>
</details>

<details>
<summary><strong>Prepare</strong></summary>

## Prepare

#### Data Preparation Process:

- Most of the financial data we encountered was relatively clean and had minimal null values.
- The null values we did encounter were mainly due to missing data that did not extend beyond 2020.

#### Handling Missing Values:

- For variables where it made sense, we filled in missing values using a 3-period weighted moving average.

- However, for variables where filling in missing data would be unreliable, we decided to drop those variables.

#### Adjust Monetary Variables
- We used the CPI method of deflating all of our monetary variables. 
    - The Formula is as follows: 2003 Price x (2023 CPI / 2003 CPI) = 2023 Price

#### Dataframe Preparation for Exploration:

1. First, we lagged the revenue back one quarter. This ensured that revenue would be predicting the quarter ahead of it. For instance, 2022 Q4 data would be used to predict 2023 Q1 revenue.

2. Next, we removed the top row, which contained data relevant to predicting Q3 2023. (Unnecessary for this study)

3. Finally, we isolated 1 line of data. Q1 2023 data was removed and set aside as a separate 1-line dataframe. This dataframe will be used to predict Q2 revenue for Ford, ATT, and Starbucks.

<a href="#top">Return to Table of Contents</a>
</details>

<details>
<summary><strong>Exploration</strong></summary>

## Exploration

#### Data Split and Model Selection:
- At the outset, we divided our data into training and test sets following a 70/30 split. As our modeling approach incorporates GridSearch, a traditional train-validate-test division was deemed unnecessary.

#### Testing for Normalcy and Statistical Methodology:
- To ensure the reliability of our analysis, we applied the Shapiro-Wilks test to examine the normality of our target variables. Notably, none of our targets exhibited a normal distribution. Acknowledging this, we opted for the utilization of appropriate statistical methods suited for non-parametric data.

#### Spearman's Rank Correlation for Continuous Variables:
- To gauge the relationships between our continuous variables and targets, we employed Spearman's rank correlation test. This rigorous examination allowed us to test each variable for significance concerning each target.

#### Data-Driven Approach to Feature Selection:
- Remaining unbiased, we allowed the data and statistical tests to guide our feature selection process. Consequently, we prepared three distinct sets of features based on what was statistically significant to each of our targets: Ford, AT&T, and Starbucks.

- By adopting this meticulous approach, we have laid a robust foundation for our predictive modeling and analysis, ensuring the accuracy and relevance of our results.

<a href="#top">Return to Table of Contents</a>
</details>

<details>
<summary><strong>Modeling</strong></summary>

## Modeling

#### Feature Selection with KBest:

To enhance our model's performance, we initially scaled the data and employed the KBest method to identify the most important features for each target:

- After KBest feature selection:
    - Ford: 15 features
    - Starbucks: 19 features
    - AT&T: 13 features

#### Regression Models and GridSearch:

Subsequently, we simultaneously employed GridSearch to explore various hyperparameters for the following regression models using the selected features from the training data:

- LassoLars
- Generalized Linear Model
- Polynomial Regression

If polynomial regression degree = 1 was chosen as the best model, this would be equivalent to Ordinary Least Squares regression.

#### Model Evaluation:

To assess model performance, we measured two key metrics on the training data: Root Mean Squared Error (RMSE) and the coefficient of determination ($R^2$):

- RMSE: The average difference between predicted and actual values.
- $R^2$: Also known as the coefficient of determination. This value represents the percentage of the variance in our target variable that is explained by our independent variables.

#### Selecting the Best Model:

We selected the model that demonstrated the lowest RMSE and the highest $R^2$ values. This top-performing model was then utilized to predict values on our test dataset.

#### Predicting the Next Quarter:

Finally, using the one-line data frame in concert with the best performing model, we made predictions for the next quarter.

<a href="#top">Return to Table of Contents</a>
</details>

<details>
<summary><strong>Nural_Network</strong></summary>

## Nural_Network

Words
- smart things
    - even smarter things

</details>

<details>
<summary><strong>Key_Findings</strong></summary>

## Key_Findings

#### Non-Normal Distribution of Targets:
- Our target variables were found to be non-normally distributed, impacting our choice of statistical methods.

#### Starbucks as DOW and S&P 500 Proxy:
- While not officially designated as a proxy, Starbucks' revenue shows a close alignment with the movement of the DOW and S&P 500, making it a 'de-facto' proxy.

#### Impact of AT&T Acquisition:
- In 2005, the merger of SBC (Southwestern Bell Corp.) and AT&T resulted in a noticeable revenue jump from $15.81B to $43.04B between April 2005 and October 2006. This acquisition was not accounted for by our independent features.

#### Revenue Comparison with Inflation Adjustment:
- In terms of inflation-adjusted dollars in the early 2000s, Ford generated more revenue dollar for dollar compared to present date.

#### Starbucks' Resilience and Recovery:
- Starbucks demonstrated resilience during COVID and was minimally affected by the 2008 Great Recession.

#### Impact of COVID and Great Recession on AT&T:
- AT&T was heavily impacted by COVID, and the 2008 Great Recession had no significant effect on the company.
    - Unfortunately, AT&T has not fully recovered since COVID.

#### Ford's Struggle with Recession and COVID:
- Ford was heavily impacted by the 2008 Great Recession and faced significant challenges during COVID due to supply chain issues. However, Ford is now approaching pre-COVID revenue numbers.

#### Correlation of Features with Companies:
- Out of 38 features, 10 were found to be correlated with all three companies, while 6 did not correlate with any of the three.

#### Successful Proof of Concept:
- Our approach and methodology have been proven successful in building a prediction model, laying a solid foundation for future work.

<a href="#top">Return to Table of Contents</a>
</details>

<details>
<summary><strong>Steps_to_Reproduce</strong></summary>

## Steps_to_Reproduce

To facilitate easy replication of our analysis, follow these steps:

#### Download the compiled CSV file from the Github repository.
- 'entire_df_ford_starbucks_att_adjusted.csv'
- Save it to your working directory.
- Read the CSV into a notebook using the following command:
    - df = pd.read_csv("name_of_file.csv")

#### Prepped Data:
- The data is already prepped, so no additional data preparation is required.

#### Utilize Provided Functions:
- Utilize the functions provided in the wrangle.py, explore.py, and modeling.py files included in the Github repository.
- These functions will assist you in various data wrangling, exploration, and modeling tasks.

#### Copy Project_Apollo_Final_Notebook:
- Make a copy of the Project_Apollo_Final_Notebook.ipynb file to start your analysis.

#### Split into Train and Test Datasets:
- Use the split_data() function provided in the explore.py file to split your data into train and test datasets using the 70/30 method.

#### Explore the Data:
- Utilize the functions provided in the explore.py file for data exploration.
- Visualize the data and conduct statistical tests to gain insights into its characteristics.

#### Scale and Model the Data:
- Use the functions from the modeling.py file to scale and model your data.
- Implement appropriate regression techniques to predict the desired outcomes.

#### Analyze Outputs and Form Conclusions:
- Carefully examine the outputs of your analysis.
- Form conclusions based on the results obtained from the modeling process.

#### Summarize with Recommendations/Next Steps:
- Summarize your findings and insights.
- Provide recommendations for further action or possible next steps based on your analysis.

By following these steps and leveraging the provided functions, you can successfully reproduce our analysis and gain valuable insights from the dataset.

<a href="#top">Return to Table of Contents</a>
</details>

<details>
<summary><strong>Summary</strong></summary>

## Summary

The goal of this project was to predict revenue for a Blue Chip Company in the next quarter. We explored the potential of economic, socio-economic, and environmental factors in predicting revenue gains and losses. Overall, the project achieved promising results in predicting revenue for Ford, ATT, and Starbucks, showcasing the potential of the selected features and models.

<a href="#top">Return to Table of Contents</a>
</details>

<details>
<summary><strong>Conclusion</strong></summary>

## Conclusion

We can conclude that our project successfully explored the use of economic, socio-economic, and environmental factors to predict revenue gains and losses for Ford, ATT, and Starbucks. Through rigorous feature selection and testing, we identified key variables and developed models that outperformed baseline predictions and produced respectable RMSE and R2 values.

For Ford, we predict a slight decrease in revenue for Q2 compared to the previous quarter. Meanwhile, the ATT model also demonstrated promising results, projecting an increase in revenue for Q2 compared to Q1. Lastly, the Starbucks model yielded positive outcomes, indicating a slight revenue increase for Q2 in comparison to Q1.

It is worth mentioning that all the models employed the LassoLars model for their predictions.

Overall, our project highlights the potential of leveraging various factors to predict revenue changes for these companies. While further refinements and validations are needed, these results offer valuable insights and opportunities for future analysis and decision-making in their respective industries.

<a href="#top">Return to Table of Contents</a>
</details>

<details>
<summary><strong>Next_Steps</strong></summary>

## Next_Steps

Based on the project findings, we can make the following recommendations and outline potential next steps:

1. Further Refinement and Validation:
   - Validate the models by comparing the predicted revenue with the actual revenue for multiple quarters to ensure consistent performance.
   - Perform additional statistical tests and analysis to validate the relationships between the selected features and revenue changes.
   - Refine the models by incorporating additional relevant variables or exploring different algorithms to improve predictive accuracy.


2. Business Impact and Decision-Making:
   - Evaluate the impact of predicted revenue changes on business operations, financial planning, and resource allocation.
   - Conduct sensitivity analysis to assess the potential outcomes under different revenue scenarios and identify areas that require strategic attention.


3. Continuous Data Collection and Feature Selection:
   - Continuously gather updated data on economic, socio-economic, and environmental factors to capture real-time market dynamics.
   - Refine the feature selection process by exploring new variables that could enhance the models' predictive power.


4. Monitor External Factors:
   - Stay updated on industry trends, regulatory changes, and market conditions that could impact the revenue of Ford, ATT, and Starbucks.
   - Monitor external factors such as consumer behavior, competitor performance, and macroeconomic indicators to capture additional insights for revenue forecasting.


5. Collaboration and Feedback:
   - Engage with domain experts and business stakeholders to gain a deeper understanding of the factors influencing revenue changes and gather valuable insights.


6. Expand to Other Companies:
   - Apply the knowledge gained from this project to predict revenue changes for other companies in the automotive, telecommunications, and food and beverage industries.
   - Adapt and refine the models for different market sectors. 

By following these recommendations and embarking on the suggested next steps, organizations can leverage data-driven revenue predictions to make informed decisions, optimize business strategies, and gain a competitive edge in the market.

<a href="#top">Return to Table of Contents</a>
</details>





<!-- #endregion -->

```python

```
