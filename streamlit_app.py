# -*- coding: utf-8 -*-
# Copyright 2024-2025 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from pathlib import Path

import altair as alt
import geopandas as gpd
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Silver Price Calculator & Sales Dashboard",
    page_icon="🥈",
    layout="wide",
)


def _find_data_file(filename: str) -> Path | None:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / filename
        if candidate.exists():
            return candidate
    return None


@st.cache_data
def load_price_data() -> pd.DataFrame:
    path = _find_data_file("historical_silver_price.csv")
    if not path:
        st.error("Missing historical_silver_price.csv in the workspace.")
        st.stop()
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"], format="%Y-%b"
    )
    df = df.sort_values("Date")
    return df


@st.cache_data
def load_state_data() -> pd.DataFrame:
    path = _find_data_file("state_wise_silver_purchased_kg.csv")
    if not path:
        st.error("Missing state_wise_silver_purchased_kg.csv in the workspace.")
        st.stop()
    return pd.read_csv(path)


@st.cache_data
def load_india_boundary() -> gpd.GeoDataFrame:
    natural_earth_url = (
        "https://naturalearth.s3.amazonaws.com/110m_cultural/"
        "ne_110m_admin_0_countries.zip"
    )
    try:
        world = gpd.read_file(natural_earth_url)
    except Exception as exc:  # pragma: no cover - network dependent
        st.error(
            "Unable to load Natural Earth boundaries. "
            "Check your internet connection and try again."
        )
        raise exc
    india = world[world["NAME"] == "India"].to_crs("EPSG:4326")
    return india


def _normalize_state(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace("&", "and")
        .replace(".", "")
        .replace("  ", " ")
    )


STATE_COORDS = {
    "Andhra Pradesh": (15.9129, 79.7400),
    "Arunachal Pradesh": (28.2180, 94.7278),
    "Assam": (26.2006, 92.9376),
    "Bihar": (25.0961, 85.3131),
    "Chhattisgarh": (21.2787, 81.8661),
    "Goa": (15.2993, 74.1240),
    "Gujarat": (22.2587, 71.1924),
    "Haryana": (29.0588, 76.0856),
    "Himachal Pradesh": (31.1048, 77.1734),
    "Jharkhand": (23.6102, 85.2799),
    "Karnataka": (15.3173, 75.7139),
    "Kerala": (10.8505, 76.2711),
    "Madhya Pradesh": (22.9734, 78.6569),
    "Maharashtra": (19.7515, 75.7139),
    "Manipur": (24.6637, 93.9063),
    "Meghalaya": (25.4670, 91.3662),
    "Mizoram": (23.1645, 92.9376),
    "Nagaland": (26.1584, 94.5624),
    "Odisha": (20.9517, 85.0985),
    "Punjab": (31.1471, 75.3412),
    "Rajasthan": (27.0238, 74.2179),
    "Sikkim": (27.5330, 88.5122),
    "Tamil Nadu": (11.1271, 78.6569),
    "Telangana": (18.1124, 79.0193),
    "Tripura": (23.9408, 91.9882),
    "Uttar Pradesh": (26.8467, 80.9462),
    "Uttarakhand": (30.0668, 79.0193),
    "West Bengal": (22.9868, 87.8550),
    "Delhi": (28.7041, 77.1025),
    "Jammu & Kashmir": (33.7782, 76.5762),
    "Ladakh": (34.2268, 77.5619),
}


@st.cache_data
def build_state_points(_state_df: pd.DataFrame) -> gpd.GeoDataFrame:
    coords = pd.DataFrame.from_dict(
        STATE_COORDS, orient="index", columns=["lat", "lon"]
    ).reset_index()
    coords = coords.rename(columns={"index": "State"})
    merged = _state_df.merge(coords, on="State", how="left")
    missing = merged[merged["lat"].isna()]["State"].tolist()
    if missing:
        st.warning(
            "Missing coordinates for: " + ", ".join(missing), icon=":material/info:"
        )
    gdf = gpd.GeoDataFrame(
        merged,
        geometry=gpd.points_from_xy(merged["lon"], merged["lat"]),
        crs="EPSG:4326",
    )
    return gdf


price_df = load_price_data()
state_df = load_state_data()
india_boundary = load_india_boundary()
state_points = build_state_points(state_df)


st.title("Silver Price Calculator & Silver Sales Analysis")
st.caption(
    "Analyze historical silver prices, estimate costs, and explore state-wise purchases."
)

""

latest_price_kg = price_df["Silver_Price_INR_per_kg"].iloc[-1]
latest_price_date = price_df["Date"].iloc[-1].strftime("%b %Y")
avg_price = price_df["Silver_Price_INR_per_kg"].mean()
min_price = price_df["Silver_Price_INR_per_kg"].min()
max_price = price_df["Silver_Price_INR_per_kg"].max()

summary_cols = st.columns(4)
summary_cols[0].metric("Latest price (INR/kg)", f"₹{latest_price_kg:,.0f}")
summary_cols[1].metric("Average price (INR/kg)", f"₹{avg_price:,.0f}")
summary_cols[2].metric("Lowest price (INR/kg)", f"₹{min_price:,.0f}")
summary_cols[3].metric("Highest price (INR/kg)", f"₹{max_price:,.0f}")

""

calculator, history = st.columns([2, 3], gap="large")

with calculator:
    st.subheader("Silver Price Calculator")
    st.write("Calculate the total cost based on weight and price per gram.")

    use_latest = st.checkbox(
        f"Use latest dataset price from {latest_price_date}", value=True
    )
    default_price_per_gram = latest_price_kg / 1000
    price_per_gram = st.number_input(
        "Current price (INR per gram)",
        min_value=0.01,
        value=float(default_price_per_gram),
        step=0.1,
        disabled=use_latest,
    )
    if use_latest:
        price_per_gram = float(default_price_per_gram)

    weight_value = st.number_input(
        "Weight", min_value=0.01, value=10.0, step=0.1
    )
    weight_unit = st.selectbox("Unit", ["grams", "kilograms"])

    weight_in_grams = weight_value * (1000 if weight_unit == "kilograms" else 1)
    total_cost_inr = weight_in_grams * price_per_gram

    st.success(f"Total cost: ₹{total_cost_inr:,.2f}")

    st.markdown("#### Currency conversion")
    currency = st.selectbox("Convert to", ["USD", "EUR", "GBP", "AED", "Other"])
    exchange_rate = st.number_input(
        "Exchange rate (1 INR = target currency)",
        min_value=0.0001,
        value=0.012,
        step=0.001,
        format="%.4f",
    )
    converted_total = total_cost_inr * exchange_rate
    st.info(f"Converted total: {currency} {converted_total:,.2f}")

with history:
    st.subheader("Historical Silver Price Chart")
    st.write("Filter by price band (INR per kg).")

    price_filter = st.radio(
        "Price band",
        [
            "All",
            "≤ 20,000 INR/kg",
            "Between 20,000 and 30,000 INR/kg",
            "≥ 30,000 INR/kg",
        ],
        horizontal=True,
    )

    filtered = price_df.copy()
    if price_filter == "≤ 20,000 INR/kg":
        filtered = filtered[filtered["Silver_Price_INR_per_kg"] <= 20000]
    elif price_filter == "Between 20,000 and 30,000 INR/kg":
        filtered = filtered[
            filtered["Silver_Price_INR_per_kg"].between(20000, 30000)
        ]
    elif price_filter == "≥ 30,000 INR/kg":
        filtered = filtered[filtered["Silver_Price_INR_per_kg"] >= 30000]

    date_min = filtered["Date"].min()
    date_max = filtered["Date"].max()
    date_range = st.slider(
        "Date range",
        min_value=date_min.to_pydatetime(),
        max_value=date_max.to_pydatetime(),
        value=(date_min.to_pydatetime(), date_max.to_pydatetime()),
    )
    filtered = filtered[
        filtered["Date"].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))
    ]

    price_chart = (
        alt.Chart(filtered)
        .mark_line(point=True)
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Silver_Price_INR_per_kg:Q", title="INR per kg"),
            tooltip=[
                alt.Tooltip("Date:T", title="Date"),
                alt.Tooltip("Silver_Price_INR_per_kg:Q", title="INR/kg", format=",.0f"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(price_chart, use_container_width=True)

""

st.subheader("State-wise Silver Sales Dashboard")

total_kg = state_df["Silver_Purchased_kg"].sum()
top_state = state_df.loc[state_df["Silver_Purchased_kg"].idxmax(), "State"]
top_value = state_df["Silver_Purchased_kg"].max()
avg_state = state_df["Silver_Purchased_kg"].mean()

dash_cols = st.columns(4)
dash_cols[0].metric("Total purchased (kg)", f"{total_kg:,.0f}")
dash_cols[1].metric("Top state", top_state)
dash_cols[2].metric("Top state (kg)", f"{top_value:,.0f}")
dash_cols[3].metric("Average per state (kg)", f"{avg_state:,.0f}")

map_col, chart_col = st.columns([2, 3], gap="large")

with map_col:
    st.markdown("#### India map (state-wise purchases)")
    base = alt.Chart(india_boundary).mark_geoshape(
        fill="#f0f2f6", stroke="white", strokeWidth=0.6
    )

    points = (
        alt.Chart(state_points)
        .mark_circle(opacity=0.8)
        .encode(
            longitude="lon:Q",
            latitude="lat:Q",
            size=alt.Size(
                "Silver_Purchased_kg:Q",
                title="Purchased (kg)",
                scale=alt.Scale(range=[40, 900]),
            ),
            color=alt.Color(
                "Silver_Purchased_kg:Q",
                title="Purchased (kg)",
                scale=alt.Scale(scheme="blues"),
            ),
            tooltip=[
                alt.Tooltip("State:N"),
                alt.Tooltip("Silver_Purchased_kg:Q", format=",.0f"),
            ],
        )
    )

    map_chart = alt.layer(base, points).project(type="mercator").properties(height=420)
    st.altair_chart(map_chart, use_container_width=True)

with chart_col:
    st.markdown("#### Top 5 states by silver purchases")
    top_states = state_df.sort_values("Silver_Purchased_kg", ascending=False).head(5)
    bar_chart = (
        alt.Chart(top_states)
        .mark_bar()
        .encode(
            x=alt.X("Silver_Purchased_kg:Q", title="Purchased (kg)"),
            y=alt.Y("State:N", sort="-x", title="State"),
            color=alt.Color(
                "Silver_Purchased_kg:Q",
                legend=None,
                scale=alt.Scale(scheme="blues"),
            ),
            tooltip=[
                alt.Tooltip("State:N"),
                alt.Tooltip("Silver_Purchased_kg:Q", format=",.0f"),
            ],
        )
        .properties(height=240)
    )
    st.altair_chart(bar_chart, use_container_width=True)

    st.markdown("#### January silver price trend")
    jan_df = price_df[price_df["Month"] == "Jan"].copy()
    jan_df = jan_df.sort_values("Year")
    jan_line = (
        alt.Chart(jan_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("Year:O", title="Year"),
            y=alt.Y("Silver_Price_INR_per_kg:Q", title="INR per kg"),
            tooltip=[
                alt.Tooltip("Year:O"),
                alt.Tooltip("Silver_Price_INR_per_kg:Q", format=",.0f"),
            ],
        )
        .properties(height=240)
    )
    st.altair_chart(jan_line, use_container_width=True)

st.markdown("#### Data preview")
preview_cols = st.columns(2)
preview_cols[0].dataframe(price_df[["Date", "Silver_Price_INR_per_kg"]], height=240)
preview_cols[1].dataframe(state_df, height=240)
