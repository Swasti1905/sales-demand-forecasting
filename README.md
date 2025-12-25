📈 Sales Demand Forecasting Web App



📌 Overview

This project is an end-to-end \*\*sales demand forecasting system\*\* built using time series analysis and machine learning techniques.  

It analyzes historical sales data, identifies trends and seasonality, forecasts future demand, and presents results through an interactive \*\*Streamlit web application\*\*.



The project is designed to support \*\*business decision-making\*\* such as inventory planning, sales forecasting, and demand management.



🗂 Dataset

The dataset contains historical sales information with the following columns:



\- `data` – Date

\- `venda` – Sales (target variable)

\- `estoque` – Inventory level

\- `preco` – Product price



The dataset is stored locally in the `data/` folder as:

data/sales\_data.csv





🛠️ Technologies Used

\- Python

\- Pandas, NumPy

\- Matplotlib

\- Statsmodels (ARIMA)

\- TensorFlow / Keras (LSTM)

\- Scikit-learn

\- Streamlit (Web App)





📊 Project Structure

sales-demand-forecasting/

│

├── data/

│ └── sales\_data.csv

│

├── notebooks/

│ ├── 01\_data\_cleaning\_and\_eda.ipynb

│ ├── 02\_trend\_and\_seasonality\_analysis.ipynb

│ ├── 03\_arima\_forecasting.ipynb

│ ├── 04\_lstm\_forecasting.ipynb

│ └── 05\_model\_comparison\_and\_insights.ipynb

│

├── app.py

├── requirements.txt

└── README.md





🔍 Methodology

1\. Data cleaning and exploratory data analysis  

2\. Trend and seasonality decomposition  

3\. Sales forecasting using \*\*ARIMA\*\*

4\. Sales forecasting using \*\*LSTM\*\*

5\. Model comparison using RMSE

6\. Deployment of forecasts via a \*\*Streamlit web app\*\*





📈 Results

\- ARIMA provides a strong baseline forecast for stable demand patterns.

\- LSTM performs better for capturing non-linear and complex sales behavior.

\- Forecasts help identify upcoming demand trends.





💡 Business Use Case

\- Inventory planning

\- Sales forecasting

\- Demand trend analysis

\- Decision support for pricing and supply chain management





🌐 Web Application

The Streamlit app allows users to:

\- View historical sales data

\- Select forecast horizon

\- Generate future sales forecasts using ARIMA or LSTM

\- Visualize predictions interactively





▶️ How to Run the Project Locally



1\. Install dependencies

bash

pip install -r requirements.txt



2\. Run the Streamlit app

bash

streamlit run app.py





The app will open in your browser at:

http://localhost:8501

