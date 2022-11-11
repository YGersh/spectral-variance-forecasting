import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet,ElasticNetCV, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from statistics import variance
import math
import librosa
import librosa.display
from sklearn.model_selection import train_test_split

#----------PREP DATA------------

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# load & prep data
df = pd.read_csv("btcusd.csv", index_col='time')
df.index = pd.to_datetime(df.index, unit='ms')
df = df[~df.index.duplicated(keep='first')]
df=df[~(df.index < '2017-01-01')]
df = df.resample('1H').pad()
btcusd=df

#add returns
btcusd['returns']=btcusd['close'].pct_change()
btcusd=btcusd.dropna(axis=0)

#----------CALCULATE WINDOWED VARIANCE------------
PREDICTION_WINDOW=4
SCALER=500
ret_list = btcusd['returns'].tolist()
prc_list=btcusd['close'].tolist()
var_list=[]
for i in range(len(ret_list)):
    try:
        if i>=PREDICTION_WINDOW:
            var_list.append(variance(ret_list[i-PREDICTION_WINDOW: i]))
        else:
            var_list.append(np.nan)
    except Exception as e:
        print("EXCEPTION OCCURED:")
        print(e)
btcusd['ret_var']= var_list
btcusd=btcusd.dropna(axis=0)

#ANNUALIZE VARIANCE:
factor = math.sqrt(365/PREDICTION_WINDOW)
btcusd['ret_var_annualized']=btcusd['ret_var']*factor

def var_ret_var_plot():
    """
    Optional function to call for visualization of
    :return:
    """
    fig, ax = plt.subplots(3, 1, figsize=(18, 5))
    ax[0].plot(btcusd['returns'], color='blue', label='returns')
    ax[0].set_ylabel('returns', fontsize=20, color='blue')
    ax[0].set_xlabel('date', fontsize=20)
    ax[0].set_title('BTCUSD Return', fontsize=23)
    ax[0].set_xlim([btcusd.index[0], btcusd.index[-1]])
    ax[0].grid()

    ax[1].plot(btcusd['ret_var'], color='blue', label='var')
    ax[1].set_ylabel('variance', fontsize=16, color='blue')
    ax[1].set_xlabel('date', fontsize=16)
    ax[1].set_title('Returns Variance', fontsize=16)
    ax[1].set_xlim([btcusd.index[0], btcusd.index[-1]])
    ax[1].grid()

    ax[2].plot(btcusd['ret_var_annualized'], color='blue', label='var')
    ax[2].set_ylabel('price [USD]', fontsize=16, color='blue')
    ax[2].set_xlabel('date', fontsize=16)
    ax[2].set_title('Return Variance Annualized', fontsize=16)
    ax[2].set_xlim([btcusd.index[0], btcusd.index[-1]])
    ax[2].grid()

    fig.tight_layout()
    plt.show()

#make specrogram
#spec=plt.specgram(btcusd['returns'].iloc[:-128], NFFT=128, noverlap=32, Fs=1, scale='dB', mode='magnitude', cmap='inferno')
print('starting spectrogram generation')
spec=np.abs(librosa.stft(btcusd['returns'].iloc[:-128].to_numpy(), n_fft=128, hop_length=96))


#----------CREATE DATAFRAME FOR REGRESSION------------
#transpose spectrogram
data = pd.DataFrame(spec).transpose()

#convert index to samples instead of frames
idx = data.index.to_numpy() #extract index
idx_samples=librosa.frames_to_samples(idx, hop_length=96, n_fft=128) #convert from "Frame" notation to "Sample"
data.index=idx_samples #reassign index

#CALCULATE FEATURES
rms = librosa.feature.rms(S=spec, hop_length=96, frame_length=128 ) #RMS Energy
features = pd.DataFrame(rms).transpose()
features= features.rename(columns={0: 'rms'})
features.index=idx_samples #reassign index

roloff_01 = librosa.feature.spectral_rolloff(S=spec, hop_length=96, n_fft=128, roll_percent=0.01, sr=1/3600).transpose()
roloff_99 = librosa.feature.spectral_rolloff(S=spec, hop_length=96, n_fft=128, roll_percent=0.99, sr=1/3600).transpose()
roloff_50 = librosa.feature.spectral_rolloff(S=spec, hop_length=96, n_fft=128, roll_percent=0.5, sr=1/3600).transpose()

features['roloff_01']=roloff_01
features['roloff_99']=roloff_99
features['roloff_50']=roloff_50
#data=data.iloc[1: , :] #drop first row due to NA
#data.index = spec[2][1:] #drop first index due to NA

def showspecs():
    """
    Plot spectrogram and derived features
    :return:
    """
    fig1, ax1 = plt.subplots()
    img1 = librosa.display.specshow(librosa.amplitude_to_db(spec,
                                                            ref=np.max),
                                    y_axis='linear', x_axis='frames', ax=ax1,
                                    hop_length=96, n_fft=128, sr=1/3600, vmin=-45)
    ax1.set_title('Returns Spectrogram (n_fft=128, hop_length=96)')
    fig1.colorbar(img1, ax=ax1, format="%+2.0f")
    # PLOT SPEC:
    fig, ax = plt.subplots(2)
    librosa.display.specshow(librosa.amplitude_to_db(spec,
                                                     ref=np.max),
                             y_axis='linear', x_axis='frames', ax=ax[0],
                             hop_length=96, n_fft=128,sr=1/3600, vmin=-45)
    ax[0].set_title('Power spectrogram')
    # fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax[0].plot(idx, roloff_01, label='Roll-off frequency (0.01)')
    ax[0].plot(idx, roloff_99, color='w', label='Roll-off frequency (0.99)')
    ax[0].plot(idx, roloff_50, color='g', label='Roll-off frequency (0.5)')
    ax[0].legend(loc='lower right')
    ax[1].plot(features['rms'], color='blue', label='var')
    ax[1].set_ylabel('RMS Energy')
    ax[1].set_xlabel('sample (hours)')
    ax[1].set_title('RMS Energy of Spectrogram', fontsize=16)
    ax[1].set_xlim([features.index[0], features.index[-1]])
    ax[1].grid()

    data.plot(subplots=True, sharex=True, use_index=True, stacked=True, legend=False,
              title="Frequency Components vs Time", sharey=True)

#----------MAKE MODEL------------

#create a lead-target dataframe
lead=data.shift(1, fill_value=0)

#direct to variance?
framed_variance=[]
framed_annualized_variance=[]

j=0
for i in idx_samples:
    try:
        target=ret_list[j: i]
        var = variance(target)
        framed_variance.append(var)
        framed_annualized_variance.append(var*factor)
        j=i
    except Exception as e:
        print("EXCEPTION OCCURED:")
        print(e)
j=0
prc_framed_variance=[]
prc_framed_annualized_variance=[]
for i in idx_samples:
    try:
        target=prc_list[j: i]
        var = variance(target)
        prc_framed_variance.append(var)
        prc_framed_annualized_variance.append(var*factor)
        j=i
    except Exception as e:
        print("EXCEPTION OCCURED:")
        print(e)
variance_df = pd.DataFrame(framed_annualized_variance)
variance_df= variance_df.rename(columns={0: 'framed_annualized_variance'})
variance_df.index=idx_samples #reassign index
variance_df['prc_framed_annualized_variance']=prc_framed_annualized_variance


#map to X y
X=pd.merge(data, features['rms'], left_index=True, right_index=True)
#X=sm.add_constant(X)
X['rms_lag1']=X['rms'].shift(-1)
y=lead
#y=variance_df.shift(1)
#model = sm.OLS.fit_regularized( method='elastic_net', alpha=1, L1_wt=0.5).fit()

X=X.iloc[1:-1 , :]
y=y.iloc[1:-1 , :]
model_list=[]


def crossval():
    """
    module used for cross validation
    Currently non-functional due to further experimentation
    :return:
    """
    alphas = [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1]
    l1s = [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14,0.16]
    for a in alphas:
        #for j in l1s:
        model = ElasticNet(alpha=a, max_iter=5000).fit(X, y)
        score = model.score(X, y)
        pred_y = model.predict(X)
        mse = mean_squared_error(y, pred_y)
        print("Alpha:{0:.4f}, R2:{1:.2f}, MSE:{2:.2f}, RMSE:{3:.2f}"
              .format(a, score, mse, np.sqrt(mse)))
    """
    for i in ytrain:
        print("REGRESSION FOR FREQUENCY COMPONENT: ",i)
        elastic = ElasticNet(alpha=0.001, l1_ratio=0.005, normalize=True).fit(xtrain, ytrain[i])
        models.append(elastic)
        ypred = elastic.predict(xtest)
        score = elastic.score(xtest, ytest[i])
        scores.append(score)
        mse = mean_squared_error(ytest[i], ypred)
        mses.append(mse)
        corr = np.corrcoef(ytest[i], ypred)
        correls.append(corr[1][0])
        print("R2:{0:.3f}, MSE:{1:.6f}, RMSE:{2:.6f}"
              .format(score, mse, np.sqrt(mse)))
    """

#MODEL TRAIN & PREDICT

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.15)


scores=[]
mses=[]
correls=[]
models=[]

predict_stft = pd.DataFrame(columns=range(100))
predict_inverse=pd.DataFrame(columns=range(96))
#for j in range(100):
XX = X.iloc[0:-100, :]
yy = y.iloc[0:-100, :]
xtest = X.iloc[-100:, :]
predict=[]
for i in y:
    print("REGRESSION FOR FREQUENCY COMPONENT: ",i)
    elastic = ElasticNet(alpha=0.001, l1_ratio=0.005, normalize=True, fit_intercept=False).fit(XX, yy[i])
    models.append(elastic)
    ypred = elastic.predict(xtest)
    predict_stft=predict_stft.append(pd.DataFrame(ypred).transpose())



#INVERSE TRANSFORM OF PREDICTION
for i in predict_stft:
    try:
        inv=librosa.istft(predict_stft[[i, i + 1]].to_numpy(), n_fft=128, hop_length=96)
        predict_inverse=predict_inverse.append(pd.DataFrame(inv).transpose())
    except KeyError:
        pass
#predict_inverse.index=np.linspace(0,99,100)
predict_inverse=predict_inverse.reset_index(drop=True)
predict_inverse=predict_inverse.transpose() #ROWS=FRAMES, COLUMNS=SAMPLES IN FRAME

#CALCULATE PREDICTED VARIANCE:
pred_var=[]
variance_df=variance_df.iloc[1:-1 , :]
test_variance = variance_df.iloc[-100:-1, :]
for i in predict_inverse:
    pred_var.append(variance(predict_inverse[i])*SCALER)

test_variance['predicted_var']=pred_var

#CREATE NAIVE STRATEGY
test_variance['naive_var']=test_variance['framed_annualized_variance'].shift(-1, fill_value=test_variance['framed_annualized_variance'].to_list()[-2]).to_numpy()

#PLOT RESULT
fig, ax= plt.subplots(1,1)
ind = test_variance.index
ax.plot(ind, test_variance['framed_annualized_variance'], label='Framed Annualized Realized Variance')
ax.plot(ind, test_variance['naive_var'],  label='Naive Prediction (truth lagged by 1)')
ax.plot(ind, test_variance['predicted_var'],  label='Model Prediction')
ax.legend(loc='lower right')

#CALCULATE R2
from sklearn.metrics import r2_score
r2_naive = r2_score(test_variance['framed_annualized_variance'], test_variance['naive_var'])
r2_model = r2_score(test_variance['framed_annualized_variance'], test_variance['predicted_var'])
print("naive R2:", r2_naive, "model R2:", r2_model)

plt.show()







#STRATEGY TEST

#strategy settings
nr_days=99*4
straddle_price=5000
up_thresh = 1
DTE=7



#compute signal indicator
test_variance['pct_change_predicted']=test_variance['predicted_var'].pct_change()


#create signal
test_variance['signal']=np.where(test_variance['pct_change_predicted'].shift(-1)>=up_thresh, 1, 0)

#create evaluation data (sync price to variance prediction)
test = btcusd[40000:49408]
indx = [test.index[0]]
for i in range(99):
    indx.append(indx[0]+pd.DateOffset(days=(i+1)*4))
test_variance.index=indx[0:99]

test = test.merge(test_variance, left_index=True, right_index=True, how='outer')
test['strat_profit']=0
prof_list=[]
for ind in test.index:
    stat=test['signal'][ind]
    if stat==1:
        price_at_entry=test['close'][ind]
        price_at_exit=test['close'][ind+pd.DateOffset(days=DTE)] #exit
        entry=straddle_price
        payoff = abs(price_at_exit-price_at_entry)
        profit=payoff-entry
        print("| Date", ind, "| BTC price at entry", price_at_entry, "| BTC price at exit", price_at_exit, "| Payoff", payoff, "| entry cost", entry, "| profit", profit )
        prof_list.append(profit)
        test['strat_profit'][ind+pd.DateOffset(days=DTE)] = profit
print('total profit', sum(prof_list))
print("HODL profit", test['close'].iloc[-2]-test['close'].iloc[0])


test['strat_cumulative_profit']=test['strat_profit'].cumsum()

test['hodl_daily_profit']=test['close']-test['close'].shift(1)
test['hodl_cumulative_profit']=test['hodl_daily_profit'].cumsum()
fig, ax= plt.subplots(1,1)
ind = test.index
ax.plot(ind, test['strat_cumulative_profit'], label='StraddleStrat cumulative return')
ax.plot(ind, test['hodl_cumulative_profit'],  label='HODL cumulative return')
ax.plot(ind, test['close'],  label='BTC price')
ax.legend(loc='upper right')
plt.title("Strategy Results (straddle_price="+str(straddle_price)+ ", up_thresh="+str(up_thresh)+ ", DTE="+str(DTE))
plt.xlabel("Date")
plt.ylabel("USD")
print('done)')
plt.show()