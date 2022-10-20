# Salary Prediction Project

## Introduction
A hypothetical data science consulting firm is working on a new project to develop a model that  predicts the salaries of jobs in a US city based on information such as the job type(i.e janitor to CEO), the job industry, the location of the job relative to the metropolis, the level of education of the worker, the type of degree held if any and the years of experience of the worker.

The goal of the project is to train a model on a data set of one million jobs containing details outlined above and use it to predict the salaries on a different data set of job posts.  The model is required to have a mean squared error(MSE) score less than 360.  

The train set consist of two separate csv files( one for the features and one for the target- salaries). The Features in data set are as follows:
- jobId - Unique job identifier
- companyId - Id for different companies
- jobtype - level of job ranging from janitor to CEO
- degree  - degree held if any
- major - major studied if any
- industry - job industry (e.g auto, finance, service)
- yearExperience - number of years experience of the worker
- milesFromMetropolis - how far the job was from the city's metropolis

The target file consist of the jobId and the Salaries

The test set is a single csv file with the features listed above