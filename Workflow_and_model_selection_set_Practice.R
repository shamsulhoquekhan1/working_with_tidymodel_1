

# Here I am practicing on workflow sets and model selection.
# We will be working on insurance.csv file that can be found on
# this link: 
# https://www.kaggle.com/datasets/mirichoi0218/insurance?resource=download

# Loading libraries and dataset: -----

library(tidyverse)
library(tidymodels)
library(vip)  # for vip function

insurance <- read_csv("insurance.csv")

# Let's have a have at our dataset: ----

head(insurance, 10)

# Let's see whether we have missing value ----

insurance %>% summarise(across(.cols = everything(), ~sum(is.na(.x))))

# We don't have any missing values or NA, so clean dataset

# Now we will split the data into train and test ------

set.seed(54666)

insurance_split <- initial_split(insurance, strata = "region") 

# Note: we are using Strata because we want to make sure all of our 4 categories
# of regions exist in our train and test data set.

train <- training(insurance_split)
test <- testing(insurance_split)


# We want to do cross validation -----

set.seed(46656)

cv_folds <- vfold_cv(train, v = 5)
cv_folds

# So, what is happening here is that the subset of the train data is going
# to train with 901 observations and then going to be compared and test on 
# 101 observations. Each time the selection will change.

insu_recipe <- recipe(
  charges ~.,
  data = train
) %>% step_dummy(region, one_hot = TRUE) %>% 
  step_log(charges)  

# see documentation for step_dummy and step_log. 
# one_hot is the part explanation here: https://stackoverflow.com/questions/71239648/tidymodels-recipes-can-i-use-step-dummy-to-one-hot-encode-the-categorical-var


Just_for_idea <- insu_recipe %>% 
  prep() %>%
  bake(new_data = NULL)



# Now we will run different models -------


lm_sp <- linear_reg() %>% 
  set_engine("lm") %>% 
  set_mode("regression")



rf_sp <- rand_forest() %>%
  set_mode("regression") %>%
  set_engine("randomForest", importance = TRUE)



knn_sp <- nearest_neighbor(neighbors = 4) %>% 
  set_mode("regression") %>% 
  set_engine("kknn")
  

  
# Now we will put everything in the workflow set -----

# With a single workflow we use add_recipe and add_model etc
# Here we are working with several models and we want to put 
# everything in a single workflow set. The commands will change
# but the idea is same.

workflow_set <- workflow_set(
  preproc = list(insu_recipe),        
  models = list(lm_sp, rf_sp, knn_sp)
)

# If we had more recipes in preproc list, then each recipe
# would be applied to each of the each model once. If we put cross =
# FALSE inside the workflow set, then this will not happen.



workflow_set

doParallel::registerDoParallel()   # parallel computing to run faster


insu_fit <- workflow_map(
  workflow_set,
  "fit_resamples",
  resamples = cv_folds,
  seed = 45985621  # Becasue our models have randomness
)


# Evaluate model fit ------


autoplot(insu_fit)


collect_metrics(insu_fit)


rank_results(insu_fit, rank_metric = "rmse", select_best = TRUE)
  
  

# Extracting the best workflow -----


workflow_final <- extract_workflow(insu_fit, id = "recipe_rand_forest") %>% 
  fit(train)

# Here we extracted the random forest section from insu_fit workflow and
# fitted with the train data

workflow_final


# Ploting variables of importance ------

workflow_final %>% extract_fit_parsnip() %>% 
  vip(geom = "col")

# From the graph, we can see that the smoker variable has more influence
# in predicting the model and it is followed by the age variable.


# Predict on the test data set -------


prediction <- predict(
  workflow_final %>% extract_fit_parsnip(),
  new_data = insu_recipe %>% prep() %>% bake(new_data = test)
)



test_pred <- cbind(test, prediction)
















