import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go

st.set_page_config(page_title="Liquor Sales Forecast", layout="wide")

st.title(" Liquor Demand Forecast and Inventory Recommendation")

# Upload or load data
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")  # Replace with your cleaned daily_consumption
    df['Date'] = pd.to_datetime(df['Date'])
    return df

data = load_data()

# Bar and Brand selectors
bars = sorted(data['Bar Name'].unique())
bar_selected = st.selectbox("Select Bar", bars)

brands = sorted(data[data['Bar Name'] == bar_selected]['Brand Name'].unique())
brand_selected = st.selectbox("Select Brand", brands)

# Filter data
pair_df = data[
    (data['Bar Name'] == bar_selected) &
    (data['Brand Name'] == brand_selected)
][['Date', 'Daily Consumption (ml)']].copy()

if pair_df.empty:
    st.warning("No data available for selected bar-brand combination.")
    st.stop()

# Rename for Prophet
prophet_df = pair_df.rename(columns={"Date": "ds", "Daily Consumption (ml)": "y"})

# Train model
model = Prophet(daily_seasonality=True, weekly_seasonality=True)
model.fit(prophet_df)

# Forecast
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Plot
st.subheader(f" Forecast for {bar_selected} - {brand_selected}")
fig = plot_plotly(model, forecast)
fig.update_layout(title="", xaxis_title="Date", yaxis_title="Predicted Consumption")
st.plotly_chart(fig, use_container_width=True)

# Inventory recommendation
def recommend_inventory(forecast_df, days=14, buffer_percent=0.10):
    future = forecast_df.tail(days)
    total_demand = future['yhat'].sum()
    buffer = total_demand * buffer_percent
    return round(total_demand + buffer, 2)

recommended_stock = recommend_inventory(forecast)

st.success(f" Recommended Inventory for Next 14 Days: **{recommended_stock} ml**")

# Rankings (optional full table)
if st.checkbox("Show Top/Bottom Bars by Forecasted Demand"):

    @st.cache_data
    def calculate_rankings(data):
        recommendations = []
        valid_pairs = data.groupby(["Bar Name", "Brand Name"]).size().reset_index().rename(columns={0: "Count"})

        for _, row in valid_pairs.iterrows():
            bar, brand = row['Bar Name'], row['Brand Name']
            df = data[(data['Bar Name'] == bar) & (data['Brand Name'] == brand)][['Date', 'Daily Consumption (ml)']].copy()
            df.rename(columns={"Date": "ds", "Daily Consumption (ml)": "y"}, inplace=True)

            try:
                m = Prophet(daily_seasonality=True, weekly_seasonality=True)
                m.fit(df)
                future = m.make_future_dataframe(periods=30)
                fc = m.predict(future)
                stock = recommend_inventory(fc)
                recommendations.append({
                    "Bar Name": bar,
                    "Brand Name": brand,
                    "Recommended Stock": stock
                })
            except:
                continue

        rec_df = pd.DataFrame(recommendations)
        bar_summary = rec_df.groupby("Bar Name")['Recommended Stock'].sum().reset_index()
        return rec_df, bar_summary.sort_values("Recommended Stock", ascending=False)

    rec_df, ranking = calculate_rankings(data)

    st.subheader(" Top 3 Bars by Forecasted Demand")
    st.dataframe(ranking.head(3))

    st.subheader(" Lowest 3 Bars by Forecasted Demand")
    st.dataframe(ranking.tail(3))

    st.download_button(" Download Recommendation ", rec_df.to_csv(index=False), file_name="inventory_recommendations.csv", mime="text/csv")