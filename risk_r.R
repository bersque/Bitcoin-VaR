library(tidyverse)
library(ggplot2)
library(e1071)
##Loading the data in and graphing it
dane <- readr::read_csv("Bitcoin_history_data.csv")

ggplot(dane, aes(x = Date, y = Close)) +
  geom_point(alpha = 0.4, color = "gray40") +
  labs(title = "Bitcoin Price",
       x = "Data",
       y = "Closing Price") +
  theme_minimal(base_size = 14)
##The price series exhibits substantial variation over time, with several periods of rapid
##price rise and sharp declines. The changes in the level of the series 
##motivate the use of returns rather than prices for statistical modelling.


##Calculating log-returns

dane <- dane %>% select(Date,Close) %>% mutate(logret = 0)

for(i in 2:nrow(dane)){
  val <- log(dane[i,2]$Close / dane[i-1,2]$Close)
  dane[i,]$logret = val
}
##Logarithmic returns are used instead of raw prices as they provide a 
##scale-independent measure of price changes and are more suitable for statistical 
##modelling. The return series fluctuates around zero and contains several large positive
##and negative observations, suggesting the presence of extreme market movements.


ggplot(dane,aes(x=Date,y=logret)) + geom_point()
ggplot(dane, aes(x = logret)) +
  geom_histogram(binwidth = 0.01, fill = "lightblue", color = "black", bins = 12) +
  labs(title = "Histogram of logarithmic returns", x = "Quantity", y = "Frequency")
##The histogram of Bitcoin logarithmic returns indicates a distribution with pronounced tails
##and a shape that differs from the normal distribution. This suggests that extreme 
##returns occur more frequently than would be expected under a Gaussian assumption.


##empirical statistics
ex_lr <- mean(dane$logret)
sd_lr <- sd(dane$logret)
skewness(dane$logret)
kurtosis(dane$logret, type = 2)  

##The positive kurtosis indicates heavier tails than those of a normal 
##distribution, implying a greater probability of extreme returns.

### Testing whether Bitcoin log returns follow a normal distribution
### H0: The log returns are normally distributed
### H1: The log returns are not normally distributed
shapiro.test(dane$logret)
qqnorm(dane$logret)
qqline(dane$logret)
##The Shapiro–Wilk test rejects the null hypothesis of normally distributed returns. 

##This result is consistent with the QQ plot, where the observations deviate substantially
##from the theoretical normal quantiles, particularly in the tails. Together with the 
##skewness and kurtosis statistics, these results indicate that the normal distribution
#is not an adequate description of the empirical return distribution.


acf(dane$logret, main = "ACF of Bitcoin Log Returns")
dane$logret2 <- dane$logret^2

acf(dane$logret2, main = "ACF of Squared Bitcoin Log Returns")

##The ACF of log returns shows little evidence of significant linear autocorrelation. 
##In contrast, the ACF of squared log returns shows significant positive autocorrelation
##across several lags, indicating volatility clustering.


###GARCH
library(rugarch)
spec <- ugarchspec(
  variance.model = list(
    model = "sGARCH",
    garchOrder = c(1, 1)
  ),
  mean.model = list(
    armaOrder = c(0, 0),
    include.mean = TRUE
  ),
  distribution.model = "norm"
)

fit <- ugarchfit(
  spec = spec,
  data = dane$logret
)
fit
coef(fit)
coef(fit)["alpha1"] + coef(fit)["beta1"] ### alfa + beta is close to 1 --> volatility is very persistant
sigma_garch <- sigma(fit)
#The estimated GARCH(1,1) parameters indicate substantial persistence in conditional 
#volatility. The value of α + β being close to one suggests that shocks to volatility 
#decay slowly over time. Consequently, periods of elevated volatility tend to persist 
#rather than disappearing immediately after a single shock.

plot(
  dane$Date,
  sigma_garch,
  type = "l",
  main = "GARCH Conditional Volatility",
  xlab = "Date",
  ylab = "Volatility"
)

## The estimated conditional volatility varies over time, with noticeable
## increases during periods of large market movements. This time-varying
## behaviour reflects volatility clustering observed in the squared returns.
## The GARCH model captures these changes in volatility by allowing current
## volatility to depend on past shocks and past volatility.

z <- residuals(fit, standardize = TRUE)
z
acf(z^2, main = "ACF of Squared Standardized Residuals")

Box.test(
  z^2,
  lag = 20,
  type = "Ljung-Box"
)
##After standardizing the residuals by the estimated conditional volatility, the remaining
##squared residuals show substantially less evidence of serial dependence. The Ljung–Box 
##test provides a formal check for remaining autocorrelation in squared standardized 
##residuals. A large p-value indicates that the null hypothesis of no autocorrelation 
##is not rejected, suggesting that the GARCH model has captured the main volatility 
#dependence present in the data.


hist(
  z,
  breaks = 50,
  main = "Standardized GARCH Residuals",
  xlab = "z"
)

qqnorm(z)
qqline(z)

infocriteria(fit)
##The Student-t specification could be considered as an extension, but I retained the
#Gaussian specification as the baseline to keep the subsequent VaR analysis consistent.

###Now we will proceed with different ways of estimating VaR


##Historical VaR
dane$logret
alfa <- 0.01
VaR_hist <- -quantile(dane$logret,probs = c(alfa))
VaR_hist
## Historical VaR is above 10%, indicating that approximately 1% of the
## historical returns were below this loss threshold. Since the estimate
## is based on the entire historical sample, it reflects extreme events
## observed over the whole period and may therefore be relatively high.


##Parametric VaR
q_norm <- qnorm(0.01, mean = ex_lr, sd = sd_lr)
VaR_param <- -q_norm
VaR_param
q_norm
VaR_param
## Parametric VaR assumes that log returns follow a normal distribution.
## Although the empirical distribution of returns deviates from normality,
## this approach provides a simple benchmark for comparison with the other
## VaR methods.


###Parametric GARCH
forecast <- ugarchforecast(fit, n.ahead = 1)
sigma_forecast <- sigma(forecast)

q_garch <- fitted(forecast) + qnorm(0.01) * sigma_forecast
VaR_garch <- -q_garch
VaR_garch
## Unlike the previous methods, GARCH VaR is conditional on the current
## volatility forecast. Therefore, the estimated VaR adapts to the current
## volatility level and can change over time.

## Out-of-sample rolling GARCH estimation and VaR backtesting

##To evaluate the model out of sample, the data is divided into a training period 
##and a test period. Rolling GARCH is then used to generate one-day-ahead 
##VaR forecasts. The model is periodically refitted, allowing 
##the parameter estimates to incorporate newly available observations.

n <- length(dane$logret)

train_size <- floor(0.8 * n)

train <- dane$logret[1:train_size]
test <- dane$logret[(train_size + 1):n]


spec <- ugarchspec(
  variance.model = list(
    model = "sGARCH",
    garchOrder = c(1, 1)
  ),
  mean.model = list(
    armaOrder = c(0, 0),
    include.mean = TRUE
  ),
  distribution.model = "norm"
)
roll <- ugarchroll(
  spec = spec,
  data = dane$logret,
  n.ahead = 1,
  forecast.length = length(test),
  refit.every = 20,
  refit.window = "recursive",
  calculate.VaR = TRUE,
  VaR.alpha = 0.01
)
roll
report(roll, type = "VaR")


## The Kupiec test does not reject the null hypothesis of correct
## unconditional coverage (p = 0.902). Although 10 VaR violations
## were observed compared with 8.3 expected violations, the difference
## is not statistically significant at the 5% level.
##
## The Christoffersen test also does not reject the null hypothesis
## (p = 0.919). This provides no evidence of dependence between
## VaR violations during the backtesting period.


N <- 1000000
set.seed(47895621)
monte_returns <- rnorm(N,mean=ex_lr,sd=sd_lr)
q_monte <- quantile(monte_returns,0.01)
VaR_monte_carlo <- -q_monte
VaR_monte_carlo
##A Monte Carlo simulation is used to generate a large number of hypothetical one-day returns under
#the estimated normal distribution. The 1% empirical quantile of the simulated returns is then used 
#to estimate VaR.
#Because the simulation uses the same normal distribution as the parametric VaR model, the Monte Carlo
#estimate is expected to be very close to the analytical normal VaR. The simulation therefore serves 
#primarily as a numerical verification of the parametric result rather than as an independent risk model.

## GARCH M-C

forecast <- ugarchforecast(
  fit,
  n.ahead = 1
)

mu_forecast <- fitted(forecast)
sigma_forecast <- sigma(forecast)

N <- 100000

set.seed(123)

z <- rnorm(N)

sim_returns <- mu_forecast + sigma_forecast * z

q_garch_mc <- quantile(sim_returns, 0.01)

VaR_garch_mc <- -q_garch_mc
VaR_garch_mc

hist(
  sim_returns,
  breaks = 100,
  main = "Monte Carlo Simulation of Returns",
  xlab = "Simulated return",
  probability = TRUE
)
abline(
  v = -VaR_garch_mc,
  lwd = 2
)

##The GARCH-based Monte Carlo simulation incorporates the forecasted conditional volatility rather
#than using a constant historical standard deviation.
##The resulting distribution represents the range of possible next-day returns conditional on the
#volatility forecast produced by the GARCH model. The 1% simulated quantile is used to obtain the 
#corresponding one-day-ahead VaR.

## The Monte Carlo VaR based on the unconditional normal distribution is
## estimated at approximately 8.1%. The GARCH-based Monte Carlo VaR is lower,
## at approximately 7.2%, because it uses the one-step-ahead conditional
## volatility forecast rather than the unconditional historical volatility.

## The difference illustrates the effect of accounting for the current
## volatility regime when estimating one-day-ahead risk.



### Conclusions

# The analysis shows that Bitcoin returns are highly variable and do not follow
# a normal distribution. The squared returns also show clear volatility
# clustering, meaning that periods of high volatility tend to be followed by
# other periods of high volatility.
#
# Because of this, a GARCH(1,1) model was used to estimate conditional
# volatility. The results show that volatility is highly persistent, so large
# changes in volatility tend to affect the following periods as well.
#
# The diagnostic tests show that the GARCH model captures a large part of the
# dependence in volatility.
#
# Several VaR methods were used, including historical, parametric, GARCH-based
# and Monte Carlo approaches. The GARCH VaR was also tested out of sample.
# At the 1% VaR level, neither the Kupiec nor the Christoffersen test rejected
# the corresponding null hypotheses.
#
# Overall, the results show that modelling changing volatility is important
# when measuring the risk of Bitcoin returns. The GARCH model allows the VaR
# estimate to change with the current level of volatility instead of assuming
# that volatility stays constant.
