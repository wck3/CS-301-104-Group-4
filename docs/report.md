# Predicting Rainfall Using Gradient Boosting and Regression
## William Kaminski, Juan Lopez, Pedro Pacheco

## Abstract
Our team focused on tackling the plight farmers face of inaccurate precipitation predictions, which could cause economic and emotional hardship. By increasing the accuracy of hourly rainfall predictions, we can help improve the agricultural harvest of our users. By increasing the quality of harvest, revenue of farmers could improve. Our problem focuses on maximizing the accuracy of these predictions with polarimetric radar measurements provided by the Kaggle competition, How Much Did It Rain? II. By using the polarimetric radar measurements collected, our team was able to generate accurate weather predictions with pre-collected data using Gradient Boost. 

Our team intends to tackle this problem through training a random subset of a large training data set. Due to lack of computing capabilities in Colab, we used our CUDA enabled GPUs to compute our prediction. We designed a Gradient Boost algorithm to train our data. ID numbers in the data were grouped together to reduce the clutter. We found a threshold at 106mm that optimized our data. The data was converted into a scalar dataset which reduced errors. Gradient Boosting was used to accurately predict precipitation amounts. The mean square error was used as the loss function in the Gradient Boosting, while mean absolute error was used for model accuracy. Different hyperparameters were tested in order to find the optimized predictor. 

We implemented the Jax numpy library hoping to improve the speed of computation of our Gradient Boost model, but saw a decrease in some cases. For key points in our algorithm, such as computing the gradient, Jax improved our computation speeds. Our goal was to optimize the parameters of the algorithm to provide the best possible prediction. After testing a variety of learning rates and estimators, the optimized parameters were found. Using the mean squared error, a learning rate of 0.01 and increasing the number of regression tree estimators from 100 to 250 provided the best results. 

## Introduction
We are Weather Insight! Our team of experts has decided to take on the challenge of reducing the uncertainty that weather prediction creates. The competition on Kaggle, How Much Did It Rain? II, has provided the guidelines and data we used for our calculations that will be discussed later in the report. The data collected by Kaggle was collected as polarimetric radar measurements with gauges at varying distances. This data consists of measurements over an hour time span at different intervals. Polarimetric radars provide horizontal and vertical wave pulses, which makes it easier to read the size of the precipitation. Flatter rain drops indicate rain, while elongated raindrops indicate ice crystals. The evaluation metric is Mean Absolute Error.

With our goal to maximize the accuracy of rain prediction, we will be able to revolutionize the agricultural lifestyle in a positive way. If farmers are able to rely on accurate predictions, crop yields and quality of these crops will increase, thus increasing production and revenue. These predictions calculated by our learning algorithms will not only benefit farmers, but society as a whole. With the quality and variety of crops available, there will be ample resources, which in turn can reduce costs at the time of purchase. Our team will not only take away the unpredictability of a rainy day, but minimize inaccuracies farmers face that could lead to bad harvests, or even worse, another Dust Bowl. 

After creating a Gradient Descent learning model, the data provided was able to generate an accurate prediction of rainfall. The model used two hyperparameters, the learning rate and the number of estimators. These parameters affected the error and the variance of the prediction points compared to the expected data points. This error was evaluated with a Mean Absolute Error function our team implemented which uses jax.numpy arrays. It was found that the learning rate of 0.01 and 250 estimators was the most efficient hyperparameters to minimize this loss. In the experiments section, examples of different parameters used can be seen on a graph.

## Related Work

## Data

## Methods

## Experiments

## Conclusion

## References
Anwar, M T, et al. “Rainfall Prediction Using Extreme Gradient Boosting.” *Journal of Physics: Conference Series*, vol. 1869, no. 1, 1 Apr. 2021, 10.1088/1742-6596/1869/1/012078.

Kunicki, Aleksander. “Rain_regression.” *Kaggle.com*, www.kaggle.com/code/aleksanderkunicki/rain-regression.

Monogioudis, Pantelis. “Boosting — Introduction to Data Science.” *Pantelis.github.io*, pantelis.github.io/data-science/aiml-common/lectures/ensemble/boosting/_index.html.

National Centers for Environmental Information. “Next Generation Weather Radar.” *National Centers for Environmental Information (NCEI)*, 22 Sept. 2020, www.ncei.noaa.gov/products/radar/next-generation-weather-radar.

National Oceanic and Atmospheric Administration. “NCEP Meteorological Assimilation Data Ingest System (MADIS).” *Madis.ncep.noaa.gov*, madis.ncep.noaa.gov/.

Sim, Aaron. “Estimating Rainfall from Weather Radar Readings Using Recurrent Neural Networks – Aaron Sim.” *Archive.org*, 21 Apr. 2016, web.archive.org/web/20160421003010/simaaron.github.io/Estimating-rainfall-from-weather-radar-readings-using-recurrent-neural-networks/.
