import numpy as np
import keras
from keras_self_attention import SeqSelfAttention
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed
from tensorflow.keras.models import load_model, Model

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from attention import Attention

import requests
import pandas as pd
import os
import json
import datetime
import arrow
import pickle 

import matplotlib.pyplot as plt
scriptDir = os.path.dirname(os.path.realpath(__file__))
import random

def main():
    stepsLookback = 100
    stepsPredict = 1
    predictCol='close'

    predictThisTicker='SPY'.upper()

    tickers = predictThisTicker
    tickers += ',SPY,RIOT,FSLR,VIAC,GNRC,MRIN,PLTR,TLRY,CLNE,CAT,XYF,ORCL,XPEV,TRCH,SPY,RAIL,NVDA,MSFT,TSLA,AMD,WFC,INTC,NET,CCL,MO,VTI,VOO,IVV,SWPPX,GLD,FNILX,VZ,F,GME,AMC,CLOV,AMZN,AAPL,GOOG,'
    tickers += 'MMM,ABT,ABBV,ABMD,ACN,ATVI,ADBE,AMD,AAP,AES,AFL,A,APD,AKAM,ALK,ALB,ARE,ALXN,ALGN,ALLE,LNT,ALL,GOOGL,GOOG,MO,AMZN,AMCR,AEE,AAL,AEP,AXP,AIG,AMT,AWK,AMP,ABC,AME,AMGN,APH,ADI,ANSS,ANTM,AON,AOS,APA,AAPL,AMAT,APTV,ADM,ANET,AJG,AIZ,T,ATO,ADSK,ADP,AZO,AVB,AVY,BKR,BLL,BAC,BK,BAX,BDX,BBY,BIO,BIIB,BLK,BA,BKNG,BWA,BXP,BSX,BMY,AVGO,BR,CHRW,COG,CDNS,CZR,CPB,COF,CAH,KMX,CCL,CARR,CTLT,CAT,CBOE,CBRE,CDW,CE,CNC,CNP,CERN,CF,CRL,SCHW,CHTR,CVX,CMG,CB,CHD,CI,CINF,CTAS,CSCO,C,CFG,CTXS,CLX,CME,CMS,KO,CTSH,CL,CMCSA,CMA,CAG,COP,ED,STZ,COO,CPRT,GLW,CTVA,COST,CCI,CSX,CMI,CVS,DHI,DHR,DRI,DVA,DE,DAL,XRAY,DVN,DXCM,FANG,DLR,DFS,DISCA,DISCK,DISH,DG,DLTR,D,DPZ,DOV,DOW,DTE,DUK,DRE,DD,DXC,EMN,ETN,EBAY,ECL,EIX,EW,EA,EMR,ENPH,ETR,EOG,EFX,EQIX,EQR,ESS,EL,ETSY,EVRG,ES,RE,EXC,EXPE,EXPD,EXR,XOM,FFIV,FB,FAST,FRT,FDX,FIS,FITB,FE,FRC,FISV,FLT,FMC,F,FTNT,FTV,FBHS,FOXA,FOX,BEN,FCX,GPS,GRMN,IT,GNRC,GD,GE,GIS,GM,GPC,GILD,GL,GPN,GS,GWW,HAL,HBI,HIG,HAS,HCA,PEAK,HSIC,HSY,HES,HPE,HLT,HOLX,HD,HON,HRL,HST,HWM,HPQ,HUM,HBAN,HII,IEX,IDXX,INFO,ITW,ILMN,INCY,IR,INTC,ICE,IBM,IP,IPG,IFF,INTU,ISRG,IVZ,IPGP,IQV,IRM,JKHY,J,JBHT,SJM,JNJ,JCI,JPM,JNPR,KSU,K,KEY,KEYS,KMB,KIM,KMI,KLAC,KHC,KR,LB,LHX,LH,LRCX,LW,LVS,LEG,LDOS,LEN,LLY,LNC,LIN,LYV,LKQ,LMT,L,LOW,LUMN,LYB,MTB,MRO,MPC,MKTX,MAR,MMC,MLM,MAS,MA,MKC,MXIM,MCD,MCK,MDT,MRK,MET,MTD,MGM,MCHP,MU,MSFT,MAA,MHK,TAP,MDLZ,MPWR,MNST,MCO,MS,MOS,MSI,MSCI,NDAQ,NTAP,NFLX,NWL,NEM,NWSA,NWS,NEE,NLSN,NKE,NI,NSC,NTRS,NOC,NLOK,NCLH,NOV,NRG,NUE,NVDA,NVR,NXPI,ORLY,OXY,ODFL,OMC,OKE,ORCL,OGN,OTIS,PCAR,PKG,PH,PAYX,PAYC,PYPL,PENN,PNR,PBCT,PEP,PKI,PRGO,PFE,PM,PSX,PNW,PXD,PNC,POOL,PPG,PPL,PFG,PG,PGR,PLD,PRU,PTC,PEG,PSA,PHM,PVH,QRVO,PWR,QCOM,DGX,RL,RJF,RTX,O,REG,REGN,RF,RSG,RMD,RHI,ROK,ROL,ROP,ROST,RCL,SPGI,CRM,SBAC,SLB,STX,SEE,SRE,NOW,SHW,SPG,SWKS,SNA,SO,LUV,SWK,SBUX,STT,STE,SYK,SIVB,SYF,SNPS,SYY,TMUS,TROW,TTWO,TPR,TGT,TEL,TDY,TFX,TER,TSLA,TXN,TXT,TMO,TJX,TSCO,TT,TDG,TRV,TRMB,TFC,TWTR,TYL,TSN,UDR,ULTA,USB,UAA,UA,UNP,UAL,UNH,UPS,URI,UHS,UNM,VLO,VTR,VRSN,VRSK,VZ,VRTX,VFC,VIAC,VTRS,V,VNO,VMC,WRB,WAB,WMT,WBA,DIS,WM,WAT,WEC,WFC,WELL,WST,WDC,WU,WRK,WY,WHR,WMB,WLTW,WYNN,XEL,XLNX,XYL,YUM,ZBRA,ZBH,ZION,ZTS'
    tickers = tickers.split(',')
    tickers = list(set(filter(None, tickers)))
    tickers.sort()
    tickerI = np.arange(0, len(tickers))
    tickerI = np.reshape(tickerI, (tickerI.shape[0], 1))

    print(tickers.index(predictThisTicker))

    # get ticker data
    #dat = get_quote_data(predictThisTicker,'5d','1m')
    dat = get_quote_data(predictThisTicker,'10y','1d')
    print(dat)

    model, scaler_tic, scaler_tiy, scaler_vol = '','','',''

    if len(tickers)==1:
        model_path = f"{scriptDir}\\lstm.{predictThisTicker}-{stepsLookback}-{stepsPredict}-{predictCol}.h5"
        scaler_path = f"{scriptDir}\\lstm.{predictThisTicker}-{stepsLookback}-{stepsPredict}-{predictCol}.scaler"
        img_path = f"{scriptDir}\\lstm.{predictThisTicker}-{stepsLookback}-{stepsPredict}-{predictCol}.png"
    else:
        model_path = f"{scriptDir}\\lstm{len(tickers)}.{stepsLookback}-{stepsPredict}-{predictCol}.h5"
        scaler_path = f"{scriptDir}\\lstm{len(tickers)}.{stepsLookback}-{stepsPredict}-{predictCol}.scaler"
        img_path = f"{scriptDir}\\lstm{len(tickers)}.{stepsLookback}-{stepsPredict}-{predictCol}.png"


    if os.path.exists(model_path) and os.path.getsize(model_path)>1:
        model = load_model(model_path)
        with open(scaler_path, 'rb') as f: 
            scaler_ti,scaler_tic,scaler_tiy,scaler_vol = pickle.load(f)

    else:
        # scale data
        scaler_ti = MinMaxScaler(feature_range=(0, 1))
        scaler_tic = MinMaxScaler(feature_range=(0, 1))
        scaler_tiy = MinMaxScaler(feature_range=(0, 1))
        scaler_vol = MinMaxScaler(feature_range=(0, 1))

        data_x, data_y = [], []

        # calculate MIN MAX
        for ticker in tickers:
            dat = get_quote_data(ticker,'10y','1d')
            scaler_tic.fit(dat[['open','high','low','close','adjclose']].values)
            scaler_tiy.fit(dat[[predictCol]].values)
            scaler_vol.fit(dat[['volume']].values)
        scaler_ti.fit(tickerI)

        # slice into time steps seq.
        for ticker in tickers:
            dat = get_quote_data(ticker,'10y','1d')
            dat_xt = scaler_tic.transform(dat[['open','high','low','close','adjclose']])
            dat_xv = scaler_vol.transform(dat[['volume']])

            # reshape ticker tokens
            tic_index = np.empty(dat_xt.shape[0])
            tic_index.fill(tickers.index(ticker))
            tic_index = np.reshape(tic_index, (dat_xt.shape[0], 1))
            tic_index = scaler_ti.transform(tic_index)

            # prepare the data
            dat_x = np.column_stack((tic_index, dat_xt, dat_xv))
            #dat_x = np.column_stack((dat_xt, dat_xv))
            dat_y = scaler_tiy.transform(dat[[predictCol]])

            # slice into time steps seq.
            for i in range(stepsLookback,dat.shape[0]-stepsPredict):
                data_x.append(dat_x[i-stepsLookback:i])
                data_y.append([dat_y[i:i+stepsPredict]])

        data_x = np.reshape(data_x, (len(data_x), data_x[-1].shape[0], data_x[-1].shape[1])) # (2447, 60, 5)
        data_y = np.reshape(data_y, (len(data_y), stepsPredict))# (2447, 10)

        # shuffle! to prevent overfitting
        data_x, data_y = shuffle(data_x, data_y)
        num_samples, time_steps, input_dim, output_dim = data_x.shape[0], data_x.shape[1], data_x.shape[2], data_y.shape[1]

        # Define/compile the model.
        model_input = Input(shape=(time_steps, input_dim))
        #x = LSTM(time_steps*5, return_sequences=True, activation = 'relu')(model_input)
        x = LSTM(time_steps*input_dim, return_sequences=True, activation = 'tanh', recurrent_activation='sigmoid')(model_input)
        x = Attention(time_steps)(x)
        #x = Dense(time_steps, activation='relu')(x)
        x = Dense(output_dim, activation='relu')(x)
        model = Model(model_input, x)
        model.compile(loss='mae', optimizer='adam')

        print(model.summary())

        # train
        callback = keras.callbacks.EarlyStopping(monitor='loss', patience=1, restore_best_weights=True)

        #model.fit(data_x, data_y, epochs=stepsLookback+stepsPredict*5)
        model.fit(data_x, data_y, epochs=30, batch_size=len(tickers), callbacks=[callback])
        #model.fit(data_x, data_y, epochs=30, batch_size=50, callbacks=[callback])
        
        # save
        model.save(model_path)
        with open(scaler_path, 'wb') as f: 
            pickle.dump((scaler_ti,scaler_tic,scaler_tiy,scaler_vol), f, pickle.HIGHEST_PROTOCOL)

    
    #################### PREDICT ####################
    graphWidth = 180



    # scalers
    dat_xt = scaler_tic.transform(dat[-1*stepsLookback:][['open','high','low','close','adjclose']])
    dat_xv = scaler_vol.transform(dat[-1*stepsLookback:][['volume']])
    
    # reshape ticker tokens
    tic_index = np.empty(dat_xt.shape[0])
    tic_index.fill(tickers.index(predictThisTicker))
    tic_index = np.reshape(tic_index, (dat_xt.shape[0], 1))
    tic_index = scaler_ti.transform(tic_index)

    dat_x = [np.column_stack((tic_index, dat_xt, dat_xv))]
    # reshape
    data_x = np.reshape(dat_x, (len(dat_x), dat_x[-1].shape[0], dat_x[-1].shape[1]))
    # predict
    pred = model.predict(data_x)
    pred = scaler_tiy.inverse_transform(pred)
    futureW=np.concatenate([dat[-1*graphWidth-stepsPredict:][predictCol].values, pred[0]])



    offset = random.randrange(10,60)
    # scalers
    dat_xt = scaler_tic.transform(dat[-1*stepsLookback-offset:-1*offset][['open','high','low','close','adjclose']])
    dat_xv = scaler_vol.transform(dat[-1*stepsLookback-offset:-1*offset][['volume']])

    # reshape ticker tokens
    tic_index = np.empty(dat_xt.shape[0])
    tic_index.fill(tickers.index(predictThisTicker))
    tic_index = np.reshape(tic_index, (dat_xt.shape[0], 1))
    tic_index = scaler_ti.transform(tic_index)

    #dat_x = [np.column_stack((np.empty(dat_xt.shape[0]).fill(tickers.index(predictThisTicker)), dat_xt, dat_xv))]
    dat_x = [np.column_stack((tic_index, dat_xt, dat_xv))]

    # reshape
    data_x = np.reshape(dat_x, (len(dat_x), dat_x[-1].shape[0], dat_x[-1].shape[1]))

    # predict
    pred = model.predict(data_x)
    pred = scaler_tiy.inverse_transform(pred)
    future = np.concatenate([dat[-1*graphWidth-stepsPredict:-1*offset][predictCol].values, pred[0]])
    #future = future[-1 * (stepsLookback * graphWidthMultiplier - stepsPredict):]




    # plot
    plt.figure(figsize=(16,8))
    plt.title(predictThisTicker)
    plt.plot(futureW, label='Future '+predictCol)
    plt.plot(future, label='Predicted '+predictCol)
    plt.plot(dat[predictCol].values[-1*(graphWidth+stepsPredict):], label='Actual '+predictCol)
    plt.text(1, 1, model.summary())
    plt.legend()
    plt.savefig(img_path)
    plt.show()









def get_quote_data(symbol='SPY', data_range='10y', data_interval='1d'):
    cf = f"{scriptDir}\\cache\\{symbol}-{data_range}-{data_interval}.txt"
    cd = ''
    if os.path.exists(cf) and os.path.getsize(cf)>1:
        with open(cf,'r') as f:
            cd = f.read()
    else:
        print(f'Downloading {symbol}')
        res = requests.get('https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={data_range}&interval={data_interval}'.format(**locals()))
        cd = res.text
        with open(cf,'w') as f:
            f.write(cd)
    data = json.loads(cd)

    body = data['chart']['result'][0]
    dt = datetime.datetime
    dt = pd.Series(map(lambda x: arrow.get(x).datetime.replace(tzinfo=None), body['timestamp']), name='datetime')
    df = pd.DataFrame(body['indicators']['quote'][0], index=dt)
    df['adjclose'] = body['indicators']['adjclose'][0]['adjclose']
    dg = pd.DataFrame(body['timestamp'])
    df = df.loc[:, ('open', 'high', 'low', 'close', 'volume', 'adjclose')]
    df.dropna(inplace=True)
    df.columns = ['open', 'high','low','close','volume', 'adjclose']

    return df



if __name__ == '__main__':
    main()

    