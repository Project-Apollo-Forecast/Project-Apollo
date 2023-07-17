# Project-Apollo
Predict Blue-Chip Quarterly Revenue

# Project Description
By using a verity of publicly available economic, political, and social features and through the use of regression modeling this project aims to predict the quarterly revenue of blue-chip companies. 3 companies were used as examples; Ford, Starbucks, and ATT.


# Project Goal
 
* To predict the quarterly revenue of blue-chip companies


# Initial Thoughts

* Social features will have the largest impact on accuracy of the model.

# The Plan
 
* Acquire data:
    * Download required csv, pdf, xlsx files from various online sources and convert into a dataframes
    * Combine into one dataframe 
    
* Prepare data:
   * Remove nulls by replacing with rolling avg
   * Offset the quarterly revenue one row down
	 


* Explore data:
   * Answer the following initial questions:
       1. Which features have the highest correlation with revenue 
       

# Data Dictionary

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

# Steps to Reproduce
1) Clone this repo 
2) Run notebook

 
# Takeaways and Conclusions<br>

* **Exploration** 
    * Targets are not normally distributed, therefore spearman's rank correlation test was used for all features
    * 20 of 38 features tested are significant to Starbucks revenue
    * 17 of 38 features tested are significant to Ford Motor Company's revenue
    * 27 of 38 features tested are significant to ATT's revenue
    * RFE and Kbest will be run to avoid 'curse of dimensionality'


* **Modeling**
    *  A regression gridsearch was run on the relevant features for each company. With the following results:
        * Starbucks:
            * Polynomial regression was the best model
            * Beat RMSE baseline of 2.08 with an RMSE of .30
            * Predicted a 2023 Q2 revenue of $8.78 billion
        * Ford:
            * Polynomial regression was the best model
            * Beat RMSE baseline of 9.46 with an RMSE of 5.33
            * Predicted a 2023 Q2 revenue of $39.99 billion   
        * ATT:
            * Polynomial regression was the best model
            * Beat RMSE baseline of 10.91 with an RMSE of 5.85
            * Predicted a 2023 Q2 revenue of $35.12 billion


# Recommendation and Next Steps

* Expand to other companies
* Explore new socio-economic variables that could enhance the models' predictive power.
* Continuously gather updated data on economic, socio-economic, and environmental factors to capture real-time market dynamics


