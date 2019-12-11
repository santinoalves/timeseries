# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 09:17:42 2019

@author: Connor Simpson
"""
import pandas as pd
import adjustTimeSeries as ats


def createFullBase2(data,classNames):
    #working with all data
    #data = [darwinAirTemperature,darwinBiochemSB,darwinBiochemSF, darwinPressure, darwinWaves,darwinWindSpeedMax,darwinWindSpeedMin  ]
    #classNames = ['darwinAirTemperature','darwinBiochemSB','darwinBiochemSF', 'darwinPressure', 'darwinWaves','darwinWindSpeedMax','darwinWindSpeedMin']
    #define the frequency
    frequency = data[0].index[1] - data[0].index[0]
    maxMinData = data[0].index[0]
    minMaxData = data[0].index[-1]
    for i in range(0,len(data)):
        #define the frequency and date ranges
        frequency = max(frequency, data[i].index[1] - data[i].index[0])
        maxMinData = max(maxMinData,data[i].index[0])
        minMaxData  = min(minMaxData,data[i].index[-1])
    
    
    length = (minMaxData - maxMinData)/frequency
    
    #prepare list of labels: 
    columns=[]
    arraySensors = pd.DataFrame()
    for i in range(0,len(data)):
        for j in range(0,len(data[i].columns)):
            k = 1
            while data[i].index[-k]> minMaxData:
                k = k +1
            z = 0
            while maxMinData > data[i].index[z]:
                z = z +1
            #print(z)
            #print(k)
            samples = round(len(data[i][z:-k])/length)
            #print(samples)
            end = len(data[i][z:-k])+1
            #print(end)
            for n in range(samples):
                print(z)
                print(n)
                print(k)
                print(end)
                print(samples)
                columns.append(classNames[i]+'_'+data[i].columns[j]+'_'+str(n))
                arraySensors[columns[-1]]=pd.Series(data[i][data[i].columns[j]][list(range(z+samples - n,end+z-n,samples))].values,index= pd.date_range(maxMinData+frequency, minMaxData, freq=frequency))
    
    print("database created with variables: ",arraySensors.columns, "data shape: ",arraySensors.shape)
    return arraySensors

def createFullBase3(data,classNames):
    #working with all data
    #data = [darwinAirTemperature,darwinBiochemSB,darwinBiochemSF, darwinPressure, darwinWaves,darwinWindSpeedMax,darwinWindSpeedMin  ]
    #classNames = ['darwinAirTemperature','darwinBiochemSB','darwinBiochemSF', 'darwinPressure', 'darwinWaves','darwinWindSpeedMax','darwinWindSpeedMin']
    #define the frequency
    frequency = data[0].index[1] - data[0].index[0]
    maxMinData = data[0].index[0]
    minMaxData = data[0].index[-1]
    for i in range(0,len(data)):
        #define the frequency and date ranges
        frequency = max(frequency, data[i].index[1] - data[i].index[0])
        maxMinDataa = max(maxMinData,data[i].index[0])
        minMaxDataa  = min(minMaxData,data[i].index[-1])
    
    for ff in range(0,len(data)):
        if frequency == data[ff].index[1] - data[ff].index[0]:
            temp1 = 1
            while data[ff].index[-temp1] - minMaxDataa > (data[ff].index[1] - data[ff].index[0])/24:
                temp1 = temp1 + 1
            minMaxData  = data[ff].index[-temp1]
            temp2 = 0 
            while maxMinDataa - data[ff].index[temp2] > (data[ff].index[1] - data[ff].index[0])/24:
                temp2 = temp2 + 1
            maxMinData = data[ff].index[temp2]
    
    length = (minMaxData - maxMinData)/frequency
    
    #prepare list of labels: 
    columns=[]
    arraySensors = pd.DataFrame()
    for i in range(0,len(data)):
        for j in range(0,len(data[i].columns)):
            k = 1
            while data[i].index[-k] - minMaxData > (data[i].index[1] - data[i].index[0])/2:
                k = k +1
            print((data[i].index[1] - data[i].index[0])/2)
            z = 0
            while maxMinData -data[i].index[z] > (data[i].index[1] - data[i].index[0])/2:
                z = z +1
            #print(z)
            #print(k)
            samples = round(len(data[i][z:-k])/length)
            #print(samples)
            end = len(data[i][z:-k])+1
            #print(end)
            for n in range(samples):
                print(z)
                print(n)
                print(k)
                print(end)
                print(samples)
                columns.append(classNames[i]+'_'+data[i].columns[j]+'_'+str(n))
                arraySensors[columns[-1]]=pd.Series(data[i][data[i].columns[j]][list(range(z+samples - n,end+z-n,samples))].values,index= pd.date_range(maxMinData+frequency, minMaxData, freq=frequency))
    
    print("database created with variables: ",arraySensors.columns, "data shape: ",arraySensors.shape)
    return arraySensors


Biochemical = [
'https://s3-ap-southeast-2.amazonaws.com/imos-data/IMOS/ANMN/QLD/DARBGF/Biogeochem_timeseries/IMOS_ANMN-QLD_CFKSTUZ_20130621T054453Z_DARBGF_FV01_DARBGF-1306-SBE-16plus-29.9_END-20140109T054453Z_C-20150401T060718Z.nc',
]


beagleBiochem = ats.readTimeSeries(['TEMP','CNDC','PRES_REL','TURB','PAR','CHLF','PSAL','DEPTH'],Biochemical,'beagleBiochem','TEMP_quality_control')


darwinBiochema = ['https://s3-ap-southeast-2.amazonaws.com/imos-data/IMOS/ANMN/NRS/NRSDAR/Biogeochem_timeseries/IMOS_ANMN-NRS_CFKSTUZ_20130621T064645Z_NRSDAR_FV01_NRSDAR-1306-SUB-SBE16plus-22_END-20140119T011452Z_C-20170829T014429Z.nc']

darwinBiochem = ats.readTimeSeries(['TEMP','CNDC','PRES_REL','TURB','PAR','CPHL','PSAL','DEPTH'],darwinBiochema,'darwinBiochem','TEMP_quality_control')


test = createFullBase2([darwinBiochem,beagleBiochem],['darwinBiochem','beagleBiochem'])

len(darwinBiochem[z:-k])


WWSH = ['https://s3-ap-southeast-2.amazonaws.com/imos-data/IMOS/ANMN/NRS/REAL_TIME/NRSDAR/sea_surface_wave_significant_height_channel_3000/2018/QAQC/IMOS_ANMN_W_20180201T000000Z_NRSDAR_FV01.nc',
        'https://s3-ap-southeast-2.amazonaws.com/imos-data/IMOS/ANMN/NRS/REAL_TIME/NRSDAR/sea_surface_wave_significant_height_channel_3000/2018/QAQC/IMOS_ANMN_W_20180301T000000Z_NRSDAR_FV01.nc',
        'https://s3-ap-southeast-2.amazonaws.com/imos-data/IMOS/ANMN/NRS/REAL_TIME/NRSDAR/sea_surface_wave_significant_height_channel_3000/2018/QAQC/IMOS_ANMN_W_20180401T000000Z_NRSDAR_FV01.nc']

WWSH2 = ats.readTimeSeries(['WWSH'],WWSH,'WWSH','WWSH_quality_control')

urla = 'https://s3-ap-southeast-2.amazonaws.com/imos-data/IMOS/ANMN/NRS/NRSDAR/Wave/IMOS_ANMN-NRS_WZ_20180210T005900Z_NRSDAR_FV01_NRSDAR-1802-SUB-Sentinel-or-Monitor-Workhorse-ADCP-17.6_END-20180228T223826Z_C-20181009T041852Z.nc'
urllib.request.urlretrieve(urla, 'dataw')
dataw = Dataset('dataw')

urlb = 'https://s3-ap-southeast-2.amazonaws.com/imos-data/IMOS/ANMN/NRS/REAL_TIME/NRSDAR/wind_speed_30min_sample_epa_guidelines_channel_2891/2018/QAQC/IMOS_ANMN_M_20180201T000000Z_NRSDAR_FV01.nc'
urllib.request.urlretrieve(urlb, 'dataw22')
dataw2 = Dataset('dataw22')



WWSH = ats.readTimeSeries(['WWSH'],[urla],'WWSH','WWSH_quality_control')
WWSH2 = WWSH
urls = ['https://s3-ap-southeast-2.amazonaws.com/imos-data/IMOS/ANMN/NRS/REAL_TIME/NRSDAR/wind_speed_30min_sample_epa_guidelines_channel_2891/2018/QAQC/IMOS_ANMN_M_20180201T000000Z_NRSDAR_FV01.nc']

WSPD = ats.readTimeSeries(['WSPD_30min'],urls,'WSPD_30min','WSPD_30min_quality_control')

ahhhh = createFullBase3([WWSH,WSPD],['WWSH','WSPD'])

ahhhh.to_csv('WWSHWSPD.csv')


WWSHurl = ['https://s3-ap-southeast-2.amazonaws.com/imos-data/IMOS/ANMN/NRS/NRSDAR/Wave/IMOS_ANMN-NRS_WZ_20180210T005900Z_NRSDAR_FV01_NRSDAR-1802-SUB-Sentinel-or-Monitor-Workhorse-ADCP-17.6_END-20180228T223826Z_C-20181009T041852Z.nc',
           'https://s3-ap-southeast-2.amazonaws.com/imos-data/IMOS/ANMN/NRS/NRSDAR/Wave/IMOS_ANMN-NRS_WZ_20180410T011900Z_NRSDAR_FV01_NRSDAR-1804-SUB-Sentinel-or-Monitor-Workhorse-ADCP-17_END-20180802T001900Z_C-20181009T012008Z.nc']

WWSHa = ats.readTimeSeries(['WWSH'],WWSHurl,'WWSH','WWSH_quality_control')

WSPDurl = ['https://s3-ap-southeast-2.amazonaws.com/imos-data/IMOS/ANMN/NRS/REAL_TIME/NRSDAR/wind_speed_30min_sample_epa_guidelines_channel_2890/2018/QAQC/IMOS_ANMN_M_20180201T000000Z_NRSDAR_FV01.nc',
           'https://s3-ap-southeast-2.amazonaws.com/imos-data/IMOS/ANMN/NRS/REAL_TIME/NRSDAR/wind_speed_30min_sample_epa_guidelines_channel_2890/2018/QAQC/IMOS_ANMN_M_20180301T000000Z_NRSDAR_FV01.nc',
           'https://s3-ap-southeast-2.amazonaws.com/imos-data/IMOS/ANMN/NRS/REAL_TIME/NRSDAR/wind_speed_30min_sample_epa_guidelines_channel_2890/2018/QAQC/IMOS_ANMN_M_20180401T000000Z_NRSDAR_FV01.nc',
           'https://s3-ap-southeast-2.amazonaws.com/imos-data/IMOS/ANMN/NRS/REAL_TIME/NRSDAR/wind_speed_30min_sample_epa_guidelines_channel_2890/2018/QAQC/IMOS_ANMN_M_20180501T000000Z_NRSDAR_FV01.nc',
           'https://s3-ap-southeast-2.amazonaws.com/imos-data/IMOS/ANMN/NRS/REAL_TIME/NRSDAR/wind_speed_30min_sample_epa_guidelines_channel_2890/2018/QAQC/IMOS_ANMN_M_20180601T000000Z_NRSDAR_FV01.nc',
           'https://s3-ap-southeast-2.amazonaws.com/imos-data/IMOS/ANMN/NRS/REAL_TIME/NRSDAR/wind_speed_30min_sample_epa_guidelines_channel_2890/2018/QAQC/IMOS_ANMN_M_20180701T000000Z_NRSDAR_FV01.nc']

WSPDa = ats.readTimeSeries(['WSPD_30min'],WSPDurl,'WSPD_30min','WSPD_30min_quality_control')


WWSHdata = createFullBase3([WWSHa,WSPDa],['WWSH','WSPD'])


WWSHdata.to_csv('WWSHWSPDa.csv')



