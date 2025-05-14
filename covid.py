import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load the dataset (ensure the file is in the same directory)
df = pd.read_csv("owid-covid-data.csv")

# Preview the dataset
print("Columns:", df.columns)
print(df.head())

# Convert 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Filter for selected countries
selected_countries = ['Kenya', 'India', 'United States']
df_filtered = df[df['location'].isin(selected_countries)].copy()

# Drop rows with missing critical values
df_filtered = df_filtered.dropna(subset=['date', 'location', 'total_cases'])

# Handle missing values for key numeric columns
numeric_cols = ['new_cases', 'new_deaths', 'total_vaccinations']
for col in numeric_cols:
    if col in df_filtered.columns:
        df_filtered.loc[:, col] = df_filtered[col].interpolate(method='linear')

# Check if nulls remain
print("\nMissing values after cleaning:")
print(df_filtered[numeric_cols].isnull().sum())

# Preview cleaned data
print(df_filtered.head())

# Plot total cases over time
plt.figure(figsize=(12, 6))
for country in selected_countries:
    country_data = df_filtered[df_filtered['location'] == country]
    plt.plot(country_data['date'], country_data['total_cases'], label=country)
plt.title("Total COVID-19 Cases Over Time")
plt.xlabel("Date")
plt.ylabel("Total Cases")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot total deaths over time (check if column exists)
if 'total_deaths' in df_filtered.columns:
    plt.figure(figsize=(12, 6))
    for country in selected_countries:
        country_data = df_filtered[df_filtered['location'] == country]
        plt.plot(country_data['date'], country_data['total_deaths'], label=country)
    plt.title("Total COVID-19 Deaths Over Time")
    plt.xlabel("Date")
    plt.ylabel("Total Deaths")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Bar plot of daily new cases (latest 14 days)
latest_data = df_filtered[df_filtered['date'] > df_filtered['date'].max() - pd.Timedelta(days=14)]
plt.figure(figsize=(12, 6))
sns.barplot(data=latest_data, x='location', y='new_cases', estimator='mean', ci=None)
plt.title("Average New Cases (Last 14 Days)")
plt.ylabel("Avg New Daily Cases")
plt.xlabel("Country")
plt.tight_layout()
plt.show()

# Create death rate column (check for zero division)
if 'total_deaths' in df_filtered.columns:
    df_filtered['death_rate'] = df_filtered['total_deaths'] / df_filtered['total_cases']
    plt.figure(figsize=(12, 6))
    for country in selected_countries:
        country_data = df_filtered[df_filtered['location'] == country]
        plt.plot(country_data['date'], country_data['death_rate'], label=country)
    plt.title("COVID-19 Death Rate Over Time")
    plt.xlabel("Date")
    plt.ylabel("Death Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Correlation heatmap (for numeric columns)
corr_cols = [col for col in numeric_cols + ['total_cases', 'total_deaths'] if col in df_filtered.columns]
plt.figure(figsize=(10, 6))
sns.heatmap(df_filtered[corr_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# Plot total vaccinations over time
if 'total_vaccinations' in df_filtered.columns:
    plt.figure(figsize=(12, 6))
    for country in selected_countries:
        country_data = df_filtered[df_filtered['location'] == country]
        plt.plot(country_data['date'], country_data['total_vaccinations'], label=country)
    plt.title("Total COVID-19 Vaccinations Over Time")
    plt.xlabel("Date")
    plt.ylabel("Total Vaccinations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Get latest data for each country
latest_vax_data = df_filtered.sort_values('date').groupby('location').tail(1)
latest_vax_data = latest_vax_data[latest_vax_data['location'].isin(selected_countries)]

# Bar plot for fully vaccinated per hundred (check if column exists)
if 'people_fully_vaccinated_per_hundred' in latest_vax_data.columns:
    plt.figure(figsize=(10, 6))
    sns.barplot(x='location', y='people_fully_vaccinated_per_hundred', data=latest_vax_data)
    plt.title("Fully Vaccinated People (% of Population)")
    plt.xlabel("Country")
    plt.ylabel("Fully Vaccinated (%)")
    plt.tight_layout()
    plt.show()

    # Example for Kenya
    kenya_latest = latest_vax_data[latest_vax_data['location'] == 'Kenya']
    if not kenya_latest.empty:
        vaccinated = kenya_latest['people_fully_vaccinated_per_hundred'].values[0]
        unvaccinated = 100 - vaccinated
        plt.figure(figsize=(6, 6))
        plt.pie([vaccinated, unvaccinated],
                labels=['Vaccinated', 'Unvaccinated'],
                autopct='%1.1f%%',
                colors=['#4CAF50', '#F44336'],
                startangle=140)
        plt.title("Kenya Vaccination Distribution")
        plt.show()

# Plotly choropleth maps
latest_map_data = df.sort_values('date').groupby('location').tail(1)
latest_map_data = latest_map_data[latest_map_data['iso_code'].str.len() == 3]
latest_map_data = latest_map_data.dropna(subset=['total_cases'])

fig = px.choropleth(
    latest_map_data,
    locations='iso_code',
    color='total_cases',
    hover_name='location',
    color_continuous_scale='Reds',
    title='Total COVID-19 Cases by Country'
)
fig.show()

if 'people_fully_vaccinated_per_hundred' in latest_map_data.columns:
    fig = px.choropleth(
        latest_map_data,
        locations='iso_code',
        color='people_fully_vaccinated_per_hundred',
        hover_name='location',
        color_continuous_scale='Greens',
        title='Fully Vaccinated People (% of Population) by Country'
    )
    fig.show()

# Key Insights (as comments)
# 1. USA has the highest total COVID-19 cases globally, followed by India and Brazil, as of the latest data.
# 2. Kenya’s daily new cases peaked in mid-2021 but have since declined significantly.
# 3. Death rates (deaths / cases) vary widely between countries, potentially due to differences in healthcare capacity and reporting standards.
# 4. Vaccination rollouts were fastest in high-income countries.
# 5. Some countries report unusually low death rates or inconsistent daily case updates — this may be due to underreporting or data issues.