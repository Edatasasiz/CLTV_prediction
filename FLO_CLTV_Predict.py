##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################

# !pip install lifetimes
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)                    # sütunları yan yana göster.
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler

###############################################################
# GÖREV 1: Veriyi Hazırlama
###############################################################

# 1. flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.
df1 = pd.read_csv("CRM_Analytics/CLTV/FLO_Case2/flo_data_20k.csv")
df = df1.copy()
df.head()

# 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit,0)


# 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" değişkenlerinin
#aykırı değerleri varsa baskılayanız.

replace_with_thresholds(df, 'customer_value_total_ever_offline')
replace_with_thresholds(df, 'customer_value_total_ever_online')

replace_with_thresholds(df, 'order_num_total_ever_online')
replace_with_thresholds(df, 'order_num_total_ever_offline')


# 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir.
# Herbir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.

df['order_num_total'] = df['order_num_total_ever_online'] + df['order_num_total_ever_offline']
df['customer_value_total'] = df['customer_value_total_ever_online'] + df['customer_value_total_ever_offline']
df.head()


# 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

df.info()

for col in df.columns:
    if "date" in col:
        df[col] = pd.to_datetime(df[col])



###############################################################
# GÖREV 2: CLTV Veri Yapısının Oluşturulması
###############################################################

# 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.
# Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.

df['last_order_date'].max()
# 2021-05-30
today_date = dt.datetime(2021, 6, 1)


# 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.
cltv_df = df[["master_id", "last_order_date", "first_order_date", "order_num_total", "customer_value_total"]]
# recency
cltv_df["recency"] = (cltv_df["last_order_date"] - cltv_df["first_order_date"]).dt.days / 7

# T, Customer tenure
cltv_df["T"] = (today_date - cltv_df["first_order_date"]).dt.days / 7

# frequency
cltv_df['frequency'] = df['order_num_total']
cltv_df = cltv_df[cltv_df["frequency"] > 1]

# monetary
cltv_df["monetary_avg"] = cltv_df["customer_value_total"] / cltv_df["frequency"]

cltv_df = cltv_df[["master_id", "recency", "T", "frequency", "monetary_avg"]]

cltv_df.head(10)

###############################################################
# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, 6 aylık CLTV'nin hesaplanması
###############################################################

# 1. BG/NBD modelini fit ediniz.
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df["frequency"],
        cltv_df["recency"],
        cltv_df["T"])

# 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.
cltv_df["exp_purchases_3_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4*3,
                                                                                   cltv_df["frequency"],
                                                                                   cltv_df["recency"],
                                                                                   cltv_df["T"])


# 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.
cltv_df["exp_purchases_6_month"] = bgf.predict(4*6,
                                               cltv_df["frequency"],
                                               cltv_df["recency"],
                                               cltv_df["T"])



# 2. Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.
ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df["frequency"],
        cltv_df["monetary_avg"])

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                       cltv_df["monetary_avg"])

cltv_df.sort_values("exp_average_value", ascending=False).head()

# 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.
cltv_df["cltv"] = ggf.customer_lifetime_value(bgf,
                                              cltv_df["frequency"],
                                              cltv_df["recency"],
                                              cltv_df["T"],
                                              cltv_df["monetary_avg"],
                                              time=6,      # 6 months
                                              freq="W",    # Frequency of T value, ["W"]eekly
                                              discount_rate=0.01)

# CLTV değeri en yüksek 20 kişiyi gözlemleyiniz.
cltv_df.sort_values("cltv", ascending=False).head(20)



###############################################################
# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması
###############################################################

# 1. 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.
# cltv_segment ismi ile atayınız.
cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])


# 2. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.

cltv_df.groupby("cltv_segment").agg({"cltv": ["mean", "min", "max"],
                                     "frequency":["mean", "min", "max", "sum"],
                                     "monetary_avg":["mean", "min", "max", "sum"],
                                     "recency":["mean", "min", "max"]})

cltv_df[["master_id", "recency", "T", "frequency", "monetary_avg", "cltv", "cltv_segment"]]

