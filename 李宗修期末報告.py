#streamlit run /Users/lizongsiou/Desktop/李宗修期末報告.py

# 導入所需的庫
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from arch import arch_model
import statsmodels.api as sm
from datetime import datetime
import mpltw
import datetime as dt
import plotly.graph_objects as go


# 設置 Streamlit 頁面的基本配置
st.set_page_config(
   page_title="豬哥亮ARCH模型",  # 設定網頁標題
   page_icon='豬哥亮.png',  # 設定網頁圖標，路徑指向本地圖標文件
   layout="wide",  # 設定頁面布局為寬屏模式
   initial_sidebar_state="expanded"  # 初始時側邊欄狀態為展開
)

# 在頁面頂部顯示一幅圖片
st.image('這網站一定讚的.png', width=800)  # 圖片路徑及其寬度

# 顯示應用的標題
st.title('股票模型分析及預測分析')



# 從台灣期貨交易所網站抓取期貨資料
url = "https://www.taifex.com.tw/cht/9/futuresQADetail"
table = pd.read_html(url)  # 使用 pandas 讀取網頁上的表格
stocks1 = table[0].iloc[:, 1:4]  # 選取表格的一部分
stocks2 = table[0].iloc[:, 5:8]  # 選取表格的另一部分
stocks1.columns = ["代號", "證券名稱", "市值佔大盤比重"]  # 為數據框設置列名
stocks2.columns = ["代號", "證券名稱", "市值佔大盤比重"]
stocks1 = stocks1.dropna()  # 去除缺失值
stocks2 = stocks2.dropna()
stocks1["代號"] = stocks1["代號"].astype(str)  # 將代號列轉換為字串
stocks2["代號"] = [str(int(stocks_代號)) for stocks_代號 in stocks2["代號"]]
stocks = pd.concat([stocks1, stocks2], axis=0)  # 合併兩個數據框
stocks = stocks.reset_index(drop=True)  # 重置索引
stocks["市值佔大盤比重"] = stocks["市值佔大盤比重"].str[:-1].astype(float)/100  # 處理並轉換數據格式
stocks["代號"] = [stocks["代號"][i]+".TW" for i in range(len(stocks))]  # 給代號加上 ".TW" 字尾
stocks["代號_證券名稱_市值佔大盤比重"] = [stocks["代號"][i] + " " + stocks["證券名稱"][i] + " " + str(round(stocks["市值佔大盤比重"][i], 6)) for i in range(len(stocks))]

# 在側邊欄中顯示圖片
st.sidebar.image('豬哥亮分析師.png', width=300)
with st.sidebar.form(key='my_form'):
    # 創建側邊欄的交互式元件
    stock_ticker_name = st.selectbox('請選擇股票代號_證券名稱_市值佔大盤比重', stocks["代號_證券名稱_市值佔大盤比重"])
    stock_symbol = stock_ticker_name.split(" ")[0]

    start_date = st.date_input('模型開始日期', value=dt.date(2020, 1, 1), format="YYYY-MM-DD")
    end_date = st.date_input('模型結束日期', value=dt.date(2021, 12, 31), format="YYYY-MM-DD")
    start_date1 = st.date_input('預測開始日期', value=dt.date(2022, 1, 1), format="YYYY-MM-DD")
    end_date1 = st.date_input('預測結束日期', value=dt.date(2023, 12, 31), format="YYYY-MM-DD")

    # 創建更多側邊欄元件來接收用戶輸入的模型參數
    mean_options = st.multiselect('選擇均值模型', ['AR', 'Constant', 'Zero', 'LS', 'ARX', 'HAR', 'HARX'], default=['AR'])
    dist_options = st.multiselect('選擇分佈', ['normal', 't', 'skewt', 'gaussian', 'studentst', 'skewstudent', 'ged', 'generalized error'], default=['normal'])
    vol_options = st.multiselect('選擇波動性', ['Constant', 'GARCH', 'ARCH', 'EGARCH', 'FIGARCH', 'APARCH', 'HARCH'], default=['GARCH'])
    max_lags = st.number_input('設定最大滯後項(Lags)', min_value=0, max_value=10, value=3)
    max_p = st.number_input('設定最大自回歸項(P)', min_value=0, max_value=10, value=3)
    max_q = st.number_input('設定最大移動平均項(Q)', min_value=0, max_value=10, value=3)
    max_o = st.number_input('設定最大不對稱項(O)', min_value=0, max_value=10, value=3)
    submit_button = st.form_submit_button(label='豬大哥幫你分析')
    
if submit_button:
    bar = st.progress(0,'豬大哥計算中')
    # 使用 yfinance 下載股票數據
    stock_data = yf.download(stock_symbol, start='2000-01-01', end=datetime.now(), auto_adjust=True)
    data_all = np.log(stock_data['Close']/stock_data['Close'].shift(1)).dropna()*100
    data1 = data_all[start_date:end_date]
    bar.progress(10,'豬大哥計算中')
    # 初始化模型選擇的參數列表
    model_bic = []
    best_lags, best_dist, best_p, best_q, best_o, best_mean = None, None, None, None, None, None
    residuals, residuals_std = [], []

    # 遍歷模型參數
    for mean in mean_options:
        for dist in dist_options:
            for vol in vol_options:
                for lags in range(max_lags + 1):
                    for p in range(max_p + 1):
                        for q in range(max_q + 1):
                            for o in [0, min(p, max_o)]:
                                # 根據用戶選擇的均值模型、分佈和波動性類型，建立 ARCH 或 GARCH 模型
                                if (p == 0 & q == 0):
                                    # 如果 p 和 q 都為 0，則使用常數波動性模型
                                    model = arch_model(data1, mean=mean, lags=lags, vol="Constant", p=p, q=q, o=o, dist=dist)
                                else:
                                    # 如果 p 或 q 不為 0，則根據用戶的選擇使用對應的波動性模型
                                    model = arch_model(data1, mean=mean, lags=lags, vol=vol, p=p, q=q, o=o, dist=dist)
                                    # 擬合模型
                                model_fit = model.fit(disp="off")
                                # 將模型的貝氏資訊準則 (BIC) 添加到列表中
                                model_bic.append(model_fit.bic)
                                # 計算模型殘差
                                residualss = model_fit.resid
                                # 計算標準化殘差
                                residuals2 = residualss / model_fit.conditional_volatility
                                # 進行 Ljung-Box 檢定
                                lb_test = sm.stats.acorr_ljungbox(residuals2.dropna(), lags=20, model_df=lags+p+q+o, return_df=True)
                                lb_test_squared = sm.stats.acorr_ljungbox((residuals2.dropna())**2, lags=20, model_df=lags+p+q+o, return_df=True)
                                # 根據 BIC 和 Ljung-Box 檢定的 p 值來判斷是否為最佳模型
                                if (model_fit.bic == np.min(model_bic)) and (lb_test["lb_pvalue"].dropna() > 0.05).all() and (lb_test_squared["lb_pvalue"].dropna() > 0.05).all():
                                    bar.progress(50,'豬大哥計算中')
                                    best_mean = mean
                                    best_p = p
                                    best_q = q
                                    best_o = o
                                    best_lags = lags
                                    best_dist = dist
                                    best_vol = vol
                                    residuals = residualss
                                    residuals_std = residuals2
    # 檢查是否找到最佳模型並顯示結
    if best_mean is not None:
        bar.progress(85,'豬大哥計算中')
        st.subheader('最佳模型參數：')
        st.write(f'均值模型: {best_mean}')
        st.write(f'滯後項: {best_lags}')
        st.write(f'分佈: {best_dist}')
        st.write(f'波動性: {best_vol}')
        st.write(f'P(自回歸項): {best_p}')
        st.write(f'Q(移動平均項): {best_q}')
        st.write(f'O(不對稱項): {best_o}')
        st.subheader('模型摘要：')
        st.write(model_fit.summary())
        
        # Ljung-Box 檢定來檢查殘差的自相關性
        lags = 20  # 設置檢定的滯後數
        # 對標準化殘差進行檢定
        lb_test1 = sm.stats.acorr_ljungbox(residuals_std, lags=lags, model_df=best_lags)
        # Ljung-Box 檢定結果的可視化
        st.subheader('Ljung-Box 檢定')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=lb_test1.index, y=lb_test1['lb_pvalue'], mode='markers', name='p-value'))
        fig.add_hline(y=0.05, line=dict(color='red', dash='dash'), annotation_text='顯著性水平：0.05')
        fig.update_layout(title='Ljung-Box 檢定', xaxis_title='lags', yaxis_title='p-value')
        st.plotly_chart(fig)
        # 根據檢定結果判斷殘差是否符合 AR 模型的假設
        if (lb_test1['lb_pvalue'].dropna() > 0.05).all():
            st.write('殘差符合 AR 模型假設。')
        else:
            st.write('殘差不符合 AR 模型假設。')

        # 對殘差平方進行 Ljung-Box 檢定以檢查波動聚集效應
        lb_test_squared1 = sm.stats.acorr_ljungbox(residuals_std**2, lags=lags, model_df=best_lags)
        st.subheader('Ljung-Box 檢定（殘差平方）')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=lb_test_squared1.index, y=lb_test_squared1['lb_pvalue'], mode='markers', name='p-value'))
        fig.add_hline(y=0.05, line=dict(color='red', dash='dash'), annotation_text='顯著性水平：0.05')
        fig.update_layout(title='Ljung-Box 檢定（殘差平方）', xaxis_title='lags', yaxis_title='p-value')
        st.plotly_chart(fig)
        # 根據檢定結果判斷殘差平方是否符合無自相關性的假設
        if (lb_test_squared1['lb_pvalue'].dropna() > 0.05).all():
            st.write('殘差平方符合 AR 模型假設。')
        else:
            st.write('殘差平方不符合 AR 模型假設。')
        bar.progress(90,'豬大哥計算中')
        
        # 預測均值和波動性
        data2 = data_all[start_date1:end_date1]  # 選取用於預測的數據
        data2_n = len(data2)  # 數據點的數量
        test_n = data2_n - 250  # 測試數據集大小
        rolling_window_size = 250  # 滾動窗口的大小
        forecast_horizon = 1  # 預測地平線
        mean_forecast = pd.DataFrame()  # 初始化均值預測的 DataFrame
        vol_forecast = pd.DataFrame()  # 初始化波動率預測的 DataFrame

        # 使用滾動窗口進行預測
        for i in range(data2_n - rolling_window_size):
            # 用選定的模型參數擬合模型
            model = arch_model(data2[i:i+rolling_window_size], mean=best_mean, lags=best_lags, vol=best_vol, p=best_p, q=best_q, o=best_o, dist=best_dist)
            model_fit = model.fit(disp='off')  # 擬合模型，不顯示擬合過程
            pred = model_fit.forecast(horizon=forecast_horizon)  # 進行預測
            mean_forecast0 = pred.mean.iloc[[0]]  # 提取均值預測
            vol_forecast0 = np.sqrt(pred.variance).iloc[[0]]  # 提取波動率預測
            mean_forecast = pd.concat([mean_forecast, mean_forecast0], axis=0)  # 累積均值預測
            vol_forecast = pd.concat([vol_forecast, vol_forecast0], axis=0)  # 累積波動率預測
            bar.progress(95,'豬大哥計算中')
            # 將預測結果合併到一個 DataFrame 中
            forecast = pd.concat([mean_forecast, vol_forecast], axis=1)
            forecast.columns = ['Mean Forecast', 'Vol_Forecast']
            forecast['Mean Forecast'] = forecast['Mean Forecast'].shift(1)  # 均值預測的偏移
            forecast['Vol_Forecast'] = forecast['Vol_Forecast'].shift(1)  # 波動率預測的偏移

        # 均值預測與波動率預測的可視化
        st.subheader('均值預測 vs 波動率預測')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast['Mean Forecast'], mode='lines', name='均值預測', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast['Vol_Forecast'], mode='lines', name='波動率預測', line=dict(color='blue'), yaxis="y2"))
        fig.update_layout(title='均值預測 vs 波動率預測', xaxis_title='Time', yaxis_title='報酬率', yaxis2=dict(title='%', overlaying='y', side='right'))
        st.plotly_chart(fig)

        # 將實際數據與預測數據合併，用於後續的比較和可視化
        data = pd.concat([data2, forecast], axis=1) 
        data.columns = ['Log_Return', 'Mean Forecast', 'Vol_Forecast']

        # 均值預測、波動率預測與實際報酬率的比較可視化
        st.subheader('均值預測 vs 波動率預測 VS 實際報酬')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Mean Forecast'], mode='lines', name='均值預測', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=data.index, y=data['Log_Return'], mode='lines', name='實際報酬率', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=data.index, y=data['Vol_Forecast'], mode='lines', name='波動率預測', line=dict(color='red'), yaxis="y2"))
        fig.update_layout(title='均值預測 vs 波動率預測 VS 實際報酬', xaxis_title='Time', yaxis_title='%', yaxis2=dict(title='%', overlaying='y', side='right'))
        st.plotly_chart(fig)

        # 計算波動性指標
        real_volatility = np.sqrt(data2**2)  # 計算真實波動性
        data_volatility = pd.concat([real_volatility, vol_forecast], axis=1)  # 將真實波動性與預測波動性合併
        data_volatility.columns = ['real_volatility', 'vol_forecast']
        data_volatility = data_volatility.dropna()  # 去除缺失值

        # 波動率預測與真實波動值的比較可視化
        st.subheader('波動率預測 vs 真實波動值')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data_volatility.index, y=data_volatility['vol_forecast'], mode='lines', name='波動率預測', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=data_volatility.index, y=data_volatility['real_volatility'], mode='lines', name='真實波動值', line=dict(color='green')))
        fig.update_layout(title='波動率預測 vs 真實波動值', xaxis_title='Time', yaxis_title='%')
        st.plotly_chart(fig)

        # 計算平均絕對誤差 (MAE)
        mae = np.mean(np.abs(data_volatility['vol_forecast'] - data_volatility['real_volatility']))  # 計算 MAE
        st.write('平均絕對誤差為：', mae)  # 顯示 MAE

        # 實際值和預測值之走勢圖
        price = stock_data['Close']  # 從股票數據中獲取收盤價
        price = price[start_date1:]  # 選取預測開始日期後的數據
        data_price = pd.concat([price, mean_forecast], axis=1)  # 將實際價格和預測均值合併
        data_price.columns = ['close', 'forecast']  # 為合併後的數據框設置列名
        data_price['close_forecast'] = data_price['close'] / np.exp(data_price['forecast'] / 100)  # 計算預測的收盤價
        data_price['close_forecast'] = data_price['close_forecast'].shift(1)  # 對預測值進行偏移

        # 繪製實際價格與預測價格的對比圖
        st.subheader('實際價格對比預測價格')
        fig2 = go.Figure()
        # 添加預測價格的曲線
        fig2.add_trace(go.Scatter(x=data_price.index, y=data_price['close_forecast'], mode='lines', name='預測價格', line=dict(color='blue')))
        # 添加實際價格的曲線
        fig2.add_trace(go.Scatter(x=data_price.index, y=data_price['close'], mode='lines', name='實際價格', line=dict(color='green')))
        # 更新圖表佈局
        fig2.update_layout(title='實際價格對比預測價格', xaxis_title='Time', yaxis_title='＄')
        # 顯示圖表
        st.plotly_chart(fig2)
        bar.progress(100,'豬大哥計算完啦！')
    else:
        bar.progress(100,'豬大哥盡力了QQ')
        st.write('豬大哥沒有找到符合條件的模型。')
    # 顯示圖片
    st.image('豬哥亮說再見.png', width=800)  # 顯示一張圖片作為頁面的結尾

