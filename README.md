# Marketing Campaign
This is a machine learning algorithm that uses a banking dataset to train a model to determine if a particular target will make use of banking services.

## Description
### Features - Bank Marketing Dataset
##### Input Variables
* __age__: (numeric)
* __job__: type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self employed','services','student','technician','unemployed','unknown')
* __marital__: marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
* __education__: (categorical: primary, secondary, tertiary and unknown)
* __default__: has credit in default? (categorical: 'no','yes','unknown')
* __housing__: has housing loan? (categorical: 'no','yes','unknown')
* __loan__: has personal loan? (categorical: 'no','yes','unknown')
* __balance__: Balance of the individual.
* __contact__: contact communication type (categorical: 'cellular','telephone')
* __month__: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
* __day__: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')

* __campaign__: number of contacts performed during this campaign and for this client (numeric, includes last contact)
* __pdays__: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
* __previous__: number of contacts performed before this campaign and for this client (numeric)
* __poutcome__: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')

##### Output variable (desired target):
 - __y__ - has the client subscribed a term deposit? (binary: 'yes','no')

## Structuring The Data
`df.info`

## Data Analysis

## Tests
