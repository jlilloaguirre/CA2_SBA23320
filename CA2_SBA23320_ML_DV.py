# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import panel as pn
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
import logging
logging.getLogger('bokeh.core.json_encoder').setLevel(logging.ERROR)
logging.getLogger('bokeh.document').setLevel(logging.ERROR)
from scipy.stats import zscore
from sklearn_extra.cluster import KMedoids
from sklearn.neighbors import NearestNeighbors
import scipy.cluster.hierarchy as sch
from fcmeans import FCM
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import panel as pn
pn.extension('plotly', sizing_mode='stretch_width')
pn.config.raw_css.append("""
    .bk.panel-widget-box {
        padding: 10px;
    }
    .bk-root .bk-input {
        font-size: 14px;
    }
""")

import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# Loading and inspecting three datasets: movies.csv, ratings.csv, and tags.csv. 
# The goal is to understand each dataset's structure and content, including dimensions, column names, and missing values. 
# This exploration helps identify relevant entities and variables for clustering analysis and supports determining a suitable business focus based on data characteristics.

# %% [markdown]
# # Datasets dictionary
# 
# - movies.csv
# 
# 
# | Column    | Type    | Description                                         |
# | --------- | ------- | --------------------------------------------------- |
# | `movieId` | Integer | Unique identifier for each movie                    |
# | `title`   | String  | Title of the movie including release year           |
# | `genres`  | String  | Pipe-separated list of genres associated with movie |
# 
# 
# - ratings.csv
# 
# 
# | Column      | Type    | Description                                                 |
# | ----------- | ------- | ----------------------------------------------------------- |
# | `userId`    | Integer | Unique identifier for each user                             |
# | `movieId`   | Integer | Foreign key linking to the `movies` dataset                 |
# | `rating`    | Float   | Rating given by a user to a movie (usually from 0.5 to 5.0) |
# | `timestamp` | Integer | Unix timestamp of when the rating was made                  |
# 
# - tags.csv
# 
# 
# | Column      | Type    | Description                                        |
# | ----------- | ------- | -------------------------------------------------- |
# | `movieId`   | Integer | Foreign key linking to the `movies` dataset        |
# | `userId`    | Integer | Unique identifier for the user who applied the tag |
# | `tag`       | String  | Free-text tag applied by the user to the movie     |
# | `timestamp` | Integer | Unix timestamp of when the tag was applied         |
# 
# 

# %%
# Checking datasets
movies = pd.read_csv('movies.csv', encoding='ISO-8859-1')
ratings = pd.read_csv('rating.csv')
tags = pd.read_csv('tags.csv', encoding='ISO-8859-1')

# Summary function
def df_summary(df, name):
    print(f"\n {name} Summary")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("Missing values:", df.isnull().sum().sum())
    display(df.head())

# Apply summary function
df_summary(movies, "Movies")
df_summary(ratings, "Ratings")
df_summary(tags, "Tags")

# %% [markdown]
# # Data Preprocessing and EDA: 

# %% [markdown]
# ## Movies Dataset Introduction
# 
# The `movies.csv` file provides a structured overview of 2,500 films, each identified by a unique `movieId`, accompanied by the film’s `title`, and a list of associated `genres`. This dataset offers a foundational view of movie content without user interaction or ratings, making it ideal for analyzing production patterns, genre trends, and the overall distribution of content across decades.
# 
# The inclusion of genre labels (often multiple per film) allows for rich feature extraction, enabling multi-dimensional analysis such as genre co-occurrence, clustering, and recommendation modelling. Additionally, the release year, which is embedded within the movie title, provides a valuable temporal component that supports time-series exploration and trend detection.
# 
# It supports real-world applications like content tagging, catalogue structuring, or the development of user-facing features such as genre filters and discovery engines.

# %%
movies

# %% [markdown]
# To make the genres column usable for clustering, this step transforms multi-category genre strings into a one-hot encoded format. Each genre becomes a separate binary column, with a value of 1 indicating movie membership in that genre. This transformation quantifies categorical content-based features in a numerical format suitable for unsupervised learning.

# %%
movies1 = movies.copy()

# %%
# One-hot encode genres
movies1 = movies1.join(movies1.pop('genres').str.get_dummies(sep='|'))

# %%
movies1.head()

# %% [markdown]
# The title column shows the movie name and its release year in parentheses (e.g., Toy Story (1995)). 
# This extracts the year as a numerical feature. Including the release year may improve clustering by considering the timing of movie production and audience preferences.

# %%
#Creating 'Year' category
movies1['year'] = movies1['title'].str.extract(r'\((\d{4})\)', expand=False).astype(float)


# %%
movies1.head()

# %%
# Extract release year from title as integer, and remove year from movie title

movies1[['year']] = movies1['title'].str.extract(r'\((\d{4})\)').astype('Int64')
movies1['title'] = movies1['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)


# %%
movies1

# %%
movies1.describe(include='all')

# %% [markdown]
# ### Genre distribution of movies
# 
# This plot is illustrating the frequancy of movie genred in the movies.csv dataset through an interactive horizontal bar plot created with Plotly Express. The genre variable is one-hot encoded, and counts are determined by summing each genre column for all movies. The choice of a horizontal layout allows for longer genre names and enhances label readability. A viridis color scale is utilized to improve interpretability by visually representing genre frequency intensity. In addition, layout adjustments like as a wider left margin and automatic label spacing, are implemented to prevent visual overlap and enhance clarity.
# 
# The visualization also helps identifying dominant content categories that resonate most with the dataset. For an online retail nbusiness operating in entertainment, this can support personalized recommendations based on popular genre clusters, inventory optimization aligned with consumer demand, or marketing segmentation (for targeting younger viewers' preferences).
# 
# The interactive format help explore and personalize the analysis for each young user, traits associated with digital-native users. The color scale and horizontal layout improve engagement and readability on modern dashboards. Finally, tooltips, hover interactivity, and zoom allow viewers to explore granular insights in a self-guided way, replicating the UX expectations of online media platforms.
# 
# 
# Key Observations:
# - Drama (1160 movies) and Comedy (about 950) are the most frequent genres.
# - Thriller, Action, and Romance follow as mid-frequency categories (around 600–650 each).
# - Genres like Film-Noir, Documentary, and Western are the least common, with under 50 titles each.
# - The genre distribution is imbalanced, which may influence clustering results and must be considered during feature scaling or dimensionality reduction.

# %%
# Genre counts
genre_counts = movies1.drop(columns=['movieId', 'title', 'year']).sum().sort_values()

# Convert to DataFrame for plotly
genre_df = genre_counts.reset_index()
genre_df.columns = ['Genre', 'Count']

# Interactive bar chart with cinema style
fig = px.bar(
    genre_df, x='Count', y='Genre', orientation='h', color='Count', color_continuous_scale='viridis',
    title='Genre Distribution of Movies')

# Layout detaails for professional look
fig.update_layout(
    title_font=dict(size=20, family='Arial'),
    font=dict(size=14),
    xaxis_title='Number of Movies',
    yaxis_title='Genre',
    xaxis=dict(showgrid=True, ticks='outside'),
    yaxis=dict(
        showgrid=False,
        automargin=True
    ),
    coloraxis_colorbar=dict(title='Movie Count'),
    template='plotly_white',
    margin=dict(l=140, r=30, t=70, b=40),
    height=600
)

fig.show()

# %% [markdown]
# ### Movie Production and Genre Evolution
# 
# This section explore how movie production trends have evolved over time using three complementary interactive visualizations. They are also specifically designed for digital-native users (18–35 years) who are accustomed to rich, interactive interfaces like those found on modern streaming platforms.
# 
# ### - First visualization
# 
# First, the dataset is grouped by year to count how many movies were released annually. `value_counts()` gives the frequency of each year, which is then sorted and reset into a tidy DataFrame for plotting. Once this is coded, a Plotly Express line chart is used for interactivity, with `markers=True` to highlight each year. The 'plotly_white' theme ensures visual clarity and modern aesthetic. 
# 
# The visual design choices are tailored to match entertainment domain dashboards with light contrast, bold colors, label rotation for readability, etc. And the unified hover (`hovermode='x unified'`) improves usability, specially for touch or mobile navigation. 
# 
# 

# %%
# Prepare data
year_counts = movies1['year'].value_counts().sort_index()
year_df = year_counts.reset_index()
year_df.columns = ['Year', 'Movie Count']

# Interactive line chart with enhanced design
fig = px.line(
    year_df,
    x='Year',
    y='Movie Count',
    markers=True,
    title='Number of Movies Released Per Year',
    template='plotly_white'
)

# Custom trace and layout
fig.update_traces(line=dict(color='dodgerblue', width=2), marker=dict(size=6, color='royalblue'))

fig.update_layout(
    font=dict(size=14, family='Arial'),
    title_font=dict(size=20),
    xaxis_title='Release Year',
    yaxis_title='Number of Movies',
    xaxis=dict(showgrid=True, tickangle=45),
    yaxis=dict(showgrid=True),
    height=500,
    margin=dict(l=60, r=30, t=60, b=60),
    hovermode='x unified'
)

fig.show()

# %% [markdown]
# There is a noticeable increase in movie production beginning in the 1980s and peaking in the late 1990s. This trend corresponds with the digital transformation of filmmaking and global distribution. The pattern also indicates a rise in data points, which are beneficial for training models based on time-based behaviors.

# %% [markdown]
# ### - Second visualization
# 
# Genres are originally stored as pipe-separated strings. They're split into lists and then "exploded" so each genre-year pair becomes its own row which is very important for accurate aggregation. Then the next step is to group the data by year and genre, calculating how many movies were produced per genre each year. Renaming the columns helps keep the plot readable.
# 
# Finally, an interactive multi-line plot illustrates how genre popularity varies over time. Using colour to represent each genre and markers for each year improves readability. This visualization provides segmentation and guides model input selection for genre-based recommendations or clustering.

# %%
# Merge movies from one dataframe with year from another
movies_temp = movies.copy()
movies_temp['year'] = movies_temp['title'].str.extract(r'\((\d{4})\)').astype('Int64')
movies_temp['genres'] = movies_temp['genres'].str.split('|')
movies_temp = movies_temp.dropna(subset=['year'])

# Explode to genre-year pairs
genre_year_df = movies_temp.explode('genres')

# Group by year and genre
genre_counts = (
    genre_year_df.groupby(['year', 'genres'])
    .size()
    .reset_index(name='Movie Count')
    .rename(columns={'year': 'Year', 'genres': 'Genre'})
)

# Plot
fig = px.line(
    genre_counts,
    x='Year',
    y='Movie Count',
    color='Genre',
    line_group='Genre',
    title='Genre Trends Over Time (Number of Movies Released Per Year)',
    markers=True,
    template='plotly_white'
)

fig.update_layout(
    font=dict(size=13),
    title_font=dict(size=20),
    height=600,
    xaxis_title='Release Year',
    yaxis_title='Number of Movies',
    margin=dict(l=60, r=20, t=60, b=60),
    hovermode='x unified',
    legend_title='Genre'
)

fig.show()


# %% [markdown]
# Drama, Comedy, and Action consistently rank highest, but other genres like Sci-Fi and Documentary show spikes in specific decades. 
# This is valuable for both temporal segmentation and targeted content marketing.

# %% [markdown]
# ### - Third Visualization
# 
# To improve clarity, only the eight most frequent genres are retained. This helps in focusing analysis without overwhelming the visual. This version reuses the same chart structure but trims the dataset. Keeping the color sequence consistent helps visual association. The clean layout supports direct comparison between genre timelines.
# 

# %%
# Filter for Top 8 genres to reduce clutter
top_genres = genre_counts['Genre'].value_counts().nlargest(8).index
filtered = genre_counts[genre_counts['Genre'].isin(top_genres)]

# Create the interactive line plot
fig = px.line(
    filtered,
    x='Year',
    y='Movie Count',
    color='Genre',
    markers=True,
    line_group='Genre',
    template='plotly_white',
    title='Top 8 Genre Trends Over Time (Movies Released Per Year)'
)

# Final styling for layout
fig.update_layout(
    font=dict(size=13),
    title_font=dict(size=20),
    height=600,
    xaxis_title='Release Year',
    yaxis_title='Number of Movies',
    margin=dict(l=60, r=20, t=60, b=60),
    hovermode='x unified',
    legend_title='Genre'
)

fig.show()


# %% [markdown]
# This chart offers an improved and simpler snapshot of genre evolution, ideal for dashboards or business review. For example, Action and Thriller gain traction after 1990s, meaning that users interests are changing.

# %% [markdown]
# ### Movie Genre Frequency Over Time Interactive Heatmap Dashboard
# 
# This section presents a dynamic heatmap dashboard using Panel and Plotly to visualize movie genre evolution over time. It offers an engaging experience for younger adults, who are commonly familiar with streaming services and interactive data applications.
# 
# This visualization focuses on genre and time dimensions. It also applies to online retail or content strategy, where understanding media trends helps informing, marketing, and product alignment decisions.
# 
# 
# First, we reshape the long-format `genre_counts` into a matrix where 'rows = genres', 'columns = years', and 'cells = number of movies per genre per year'. This is the required format for  `px.imshow()`. Missing values are replaced with 0 using  `fillna(0)` to ensure a valid matrix without gaps.
# 
# The design rationale for this plot is that younger users are adapted to interactive filtering. The slider of the plot lets them focus on specific decades or time windows, aligning with habits developed through interfaces similar to Netflix, etc. 
# 
# 
# 
# 
# The 'width = 500' ensures the widget remains visually balanced in the layout. The callback function 'plot_heatmap' dynamically slices the heatmap columns based on the selected year range (yr). `px.imshow` is then used to render the data as a 2D grid. The key parameters modified from default are:
# 
# - `aspect='auto'`: allowing flexibility in width/height ratio for long timeseries.
# 
# - `color_continuous_scale='viridis'`: perceptually uniform scale, ideal for intensity-based data.
# 
# - `labels=dict(color='Movie Count')`: improves hover tooltips and color bar labeling.
# 
# - `title`: keeps UX consistent with the overall dashboard theme.
# 
# In regards to the figure design extra adjustments, the following parameters have been selected:
# 
# - `font=dict(size=13)`: aligns with the dashboard’s visual hierarchy.
# 
# - `height=600`: allows enough vertical space for all genres.
# 
# - `margin=dict(l=60, r=30, t=60, b=60)`: carefully adjusted to reduce dead space without crowding.
# 
# - `coloraxis_colorbar=dict(title='Number of Movies')`: renamed for better semantic clarity in ML/business contexts.
# 
# - `pn.pane.Plotly`: provide Plotly objects inside Panel apps.
# 
# - `responsive=True`: ensures mobile/tablet compatibility.
# 
# - `sizing_mode='stretch_width'`: allows full-width embedding for modern layouts.
# 
# Finally, to ensure that the widget updates from the user when changing the slider automatically trigger the function, `n.bind()` keeps the application reactive without requiring manual callbacks. This part was done after found some errors after changing the slider in the Jupyter Notebook. To help improve the user's understanding of the visualization and composition logic, `pn.Column()` stacks elements vertically, and markdown headers and helper text improve readability.
# 
# The dashbord has been deployed in Jupyter environment thanks to using `.servable()` inside `pn.extension()`.

# %%
# Initializes Panel with Plotly backend enabled. Required to render Plotly figures inside Panel dashboards.
pn.extension('plotly')

# Pivot for heatmap
heatmap = genre_counts.pivot(index='Genre', columns='Year', values='Movie Count').fillna(0)

# Year slider widget
slider = pn.widgets.IntRangeSlider(
    name='Select Year Range',
    start=heatmap.columns.min(),
    end=heatmap.columns.max(),
    value=(heatmap.columns.min(), heatmap.columns.max()),
    width=500
)

# Heatmap plot function
def plot_heatmap(yr):
    fig = px.imshow(
        heatmap.loc[:, yr[0]:yr[1]],
        aspect='auto',
        color_continuous_scale='viridis',
        labels=dict(color='Movie Count'),
        title='Movie Genre Frequency by Year'
    )
    fig.update_layout(
        font=dict(size=13),
        height=600,
        xaxis_title='Year',
        yaxis_title='Genre',
        margin=dict(l=60, r=30, t=60, b=60),
        coloraxis_colorbar=dict(title='Number of Movies')
    )
    return pn.pane.Plotly(fig, config={'responsive': True}, sizing_mode='stretch_width')

# Bind safely
plot = pn.bind(plot_heatmap, slider)

# Dashboard Layout
genre_heatmap_dashboard = pn.Column(
    "# Genre-Year Movie Heatmap",
    "Use the slider below to explore how movie genres evolved over time:",
    slider,
    plot
)

#dashboard.show()

# %% [markdown]
# This heatmap provides time-series trend data that is ideal for unsupervised learning. It is also exportable and intercative, making it practival for stakeholder presentations in media or retail analytics, and enables strategic insights that can informa release strategies, seasonal product recommendations, etc. 
# The visualization is exportable, interactive and adapted to diverse screen sizes. Can be adapted to multipage dashboards, converted to web apps, or integrated with ML model outputs.
# 
# It illustrates the evolution of movie genre production from 1922 to 2014. Movie numbers surged post-1980, particularly in Drama, Comedy, Action, and Thriller genres. The ‘viridis’ colour scale emphasizes peak activity periods, enhancing visual understanding of genre popularity over time. 
# From 1973 to 2012, aligning with the media consumption of digital-native users (18–35), we see growth in Action and Sci-Fi genres, reflecting trends in blockbuster cinema and tech-driven storytelling. 
# 
# Sometimes an `UnknownReferenceError` appears in Jupyter Notebook due to how interactive widgets update behind the scenes. This doesn’t affect how the dashboard works when exported or shared — the visualisation still functions correctly and looks as intended.

# %% [markdown]
# --------------------------------------------------------------

# %% [markdown]
# ## Ratings Dataset Introduction
# 
# 
# The ratings.csv file contains 264505 user-generated ratings across a large movie catalog. Each record links a user to a movie and includes a numerical rating and a timestamp. 
# The data structure enables in-depth analysis of behavioural trends like ratings over time, user activity patterns, and movie popularity. Including `timestamp` allows insights into user engagement timing, essential for understanding consumption cycles and peak engagement periods across years or genres.
# 
# In online retail and digital streaming aimed at younger adults (18–35), this data provides a foundation for interactive dashboards, personalized recommendations, and targeted content strategies. From a technical perspective, preprocessing steps such as converting Unix timestamps to human-readable dates, aggregating rating counts and averages, and extracting temporal features are essential for unlocking this dataset's full analytical and business value.
# 
# 
# 

# %%
ratings

# %%
ratings.shape

# %%
ratings.isna().sum()

# %%
ratings.describe()

# %%
# Check unique users and movies
ratings['userId'].nunique(), ratings['movieId'].nunique()


# %%
ratings1 = ratings.copy()

# %%
# Convert timestamp to readable datetime
ratings1['timestamp'] = pd.to_datetime(ratings1['timestamp'], unit='s')
ratings1

# %%
# Extracting datetime features
ratings1['year'] = ratings1['timestamp'].dt.year
ratings1['month'] = ratings1['timestamp'].dt.month
ratings1['dayofweek'] = ratings1['timestamp'].dt.day_name()

# %%
ratings1

# %% [markdown]
# ### Adding Average Rating Per Movie
# 
# A new column is added to the `ratings1` dataset to show each movie's average rating across all users, using the `groupby().transform('mean')` method. This maintains the original structure and allows for comparing individual user ratings with the general consensus, aiding future visualizations and recommendation logic.
# 
# 

# %%
# Add a new column: average rating for the movie associated with each row
ratings1['avg_rating_movie'] = ratings1.groupby('movieId')['rating'].transform('mean')

# %%
ratings1.describe()

# %% [markdown]
# The ratings dataset captures 264,505 user interactions from 1997 to 2015. The average rating is 3.50, with a tight spread between the 25% and 75% percentiles (3.00–4.00), indicating a positive skew common in user-generated ratings. This mirrors a trend where users tend to rate items they feel strongly about, especially when experiences are favourable, as seen on platforms like Netflix.
# 
# The newly engineered feature ‘avg_rating_movie’ enhances the dataset's readiness for clustering, recommendation, and content-based filtering by capturing each movie’s reputation across users. Temporal features (year, month) expose a surge in activity during the mid-2000s, aligning with the growth of algorithmic platforms and increasing digital engagement.
# 

# %% [markdown]
# ### Average Rating by Day of the Week
# 
# This visualization explores how users rate movies depending on the day of the week. Ratings have been grouped by weekday using `groupby('dayofweek')['rating'].mean()` and then explicitly ordered from Monday to Sunday using `.reindex(week_order)` to ensure correct chronological alignment on the x-axis.
# 
# The result is displayed using a horizontal interactive bar chart, styled with:
# - `viridis` color scale: chosen for perceptual clarity and consistency across the project.
# - `plotly_white`: template for a clean, report-ready aesthetic.
# - `font.size=13`: improved legibility.
# 
# This type of temporal breakdown is useful for both business and ML applications: it may guide strategic content releases. For example, higher-rated content on weekends, or be used to engineer time-based features for clustering and recommendation systems in the analysis future steps.

# %%
# Average rating by day of the week (ordered)
week_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
ratings_week = ratings1.groupby('dayofweek')['rating'].mean().reindex(week_order).reset_index()

# Plotly Express
fig = px.bar(
    ratings_week,
    x='dayofweek',
    y='rating',
    title='Average Rating by Day of the Week',
    labels={'dayofweek': 'Day of Week', 'rating': 'Average Rating'},
    color='rating',
    color_continuous_scale='viridis',
    template='plotly_white'
)

fig.update_layout(font=dict(size=13), height=450)
fig.show()


# %% [markdown]
# The bar chart reveals slight fluctuations in average movie ratings across the week. Notably, Sunday (3.53) and Monday (3.52) exhibit the highest average ratings, while Thursday records the lowest (3.47). This pattern may suggest variations in viewer sentiment or engagement, with weekends and early-week days potentially associated with more focused or reflective viewing behaviors.
# 
# It is also important to note that While the palette 'viridis' is perceptually uniform and excellent for some scientific contexts, its full range (deep purple to yellow-green) can feel too intense or distracting in user-facing dashboards, especially when fine differences are important.

# %% [markdown]
# #### Activity Ratings Dashboard (Month/Weekday)
# 
# 
# This Panel Dashboard displays two visualizations: the average rating by day of the week and the total number of ratings per month. The logic begins by grouping and reindexing the data with `groupby('dayofweek')['rating'].mean()` and a manual weekday order to ensure consistent x-axis alignment. For monthly counts, `value_counts().sort_index()` has been used to retain chronological order.
# 
# A custom colour scale has been implemented by combining `px.colors.sequential.Viridis[2:10]` with `OrRd[1:7]`, softening the high contrast of the original viridis palette. The custom colour scale improves perceptual clarity by reducing contrast while maintaining visual distinction between values. This adjustment helps represent differences in average ratings and enhances audience engagement. Transitioning from cooler to warmer tones, culminating in reds for higher values, creates a psychologically intuitive mapping: higher ratings feel more “active” or “positive.” In contrast, lower values seem cooler and quiet. This allows viewers to quickly interpret and compare rating intensity across categories.
# 
# 
# Layout settings have been explicitly configured: `font=dict(size=13)` improves readability, and `sizing_mode='stretch_width'` ensures that the plots adapt seamlessly across screen sizes. Each chart includes labeled axes and a uniform height, providing a consistent and presentation-ready format. To improve visual differentiation between similar values, the y-axis range has been manually narrowed to [3.46, 3.54], making small differences in average ratings more noticeable.
# 
# These choices enhance usability and prepare the visualizations for integration into the final multi-panel dashboard.

# %%
pn.extension('plotly')
# Custom softer viridis scale
custom_scale = px.colors.sequential.Viridis[2:10] + px.colors.sequential.OrRd[1:7]

fig_day = px.bar(
    ratings_week,
    x='dayofweek',
    y='rating',
    title='Average Rating by Day of the Week',
    color='rating',
    color_continuous_scale=custom_scale,
    labels={'dayofweek': 'Day of Week', 'rating': 'Average Rating'},
    template='plotly_white'
)
fig_day.update_layout(
    xaxis_title='Day of Week',
    yaxis_title='Average Rating',
    font=dict(size=13),
    height=450,
    yaxis=dict(range=[3.46, 3.54])
)

# Plot Monthly rating counts
monthly_counts = ratings1['month'].value_counts().sort_index().reset_index()
monthly_counts.columns = ['Month', 'Rating Count']

fig_month = px.bar(
    monthly_counts,
    x='Month',
    y='Rating Count',
    title='Number of Ratings per Month',
    color='Rating Count',
    color_continuous_scale=custom_scale,
    template='plotly_white'
)
fig_month.update_layout(
    xaxis_title='Month',
    yaxis_title='Number of Ratings',
    font=dict(size=13),
    height=450
)

# Panel Dashboard Layout
ratings_overview_dashboard = pn.Column(
    "# Ratings Data Overview",
    "## **This interactive dashboard explores user rating behavior over time, including average ratings by weekday and monthly rating volume:**",
    pn.pane.Plotly(fig_day, config={'responsive': True}, sizing_mode='stretch_width'),
    pn.pane.Plotly(fig_month, config={'responsive': True}, sizing_mode='stretch_width')
)

#dashboard.show()

# %% [markdown]
# The visualization shows moderate seasonal variation in user engagement throughout the year. Ratings peak in January (25,137), followed by December (23,705) and July (23,700). These peaks likely correspond to increased leisure time, such as post-holiday downtime, mid-year vacations, etc., when users engage more with media platforms.
# 
# On the other hand, activity dips during October (19,263) and May (19,427), which may align with work or academic cycles, such as exam periods or pre-holiday fatigue.
# The colour mapping, improved with a sophisticated warm-to-cool palette, clearly conveys engagement levels. Taller bars that shift into warmer shades like orange or red indicate peak months, whereas cooler colours represent lower participation. This clear visual coding facilitates quick understanding and enhances opportunities for business uses, including seasonal content planning and dynamic recommendation scheduling.
# 

# %% [markdown]
# #### Yearly Ratings Analysis
# 
# This visualization provides an interactive summary of user activity over time, allowing the user to switch between two key metrics: 'Rating Count' and 'Average Rating' per year. The dropdown menu (`pn.widgets.Select`) enables smooth switching between views while maintaining `year` as the x-axis, offering consistent temporal context.
# 
# To prepare the data, two separate aggregations have been performed using `groupby('year')`: one to calculate the total number of ratings (`.size()`), and another to compute the mean rating for each year. These have been then merged into a single DataFrame to support dynamic metric selection.
# 
# The `plot_by_metric()` function uses Plotly Express to generate the bar chart based on the selected metric. The use of `color=metric` and a customized color scale (`custom_scale`) reinforces the magnitude of values visually. Manual layout adjustments such as font sizing, axis titles, and `sizing_mode='stretch_width'`, that ensure clarity and responsiveness in dashboard environments.
# 
# This flexible design not only supports exploratory analysis but also prepares the ground for modeling or seasonality analysis in future machine learning steps.

# %%

pn.extension('plotly')

# Aggregate data
rating_count_by_year = ratings1.groupby('year').size().reset_index(name='Rating Count')
avg_rating_by_year = ratings1.groupby('year')['rating'].mean().reset_index(name='Average Rating')

# Merge into one DataFrame
ratings_yearly = rating_count_by_year.merge(avg_rating_by_year, on='year')

# Dropdown widget
y_selector = pn.widgets.Select(
    name='Select Metric',
    options=['Rating Count', 'Average Rating'],
    value='Rating Count',
    width=250
)

# Step 4: Reactive plotting function
def plot_by_metric(metric):
    fig = px.bar(
        ratings_yearly,
        x='year',
        y=metric,
        title=f'{metric} per Year',
        color=metric,
        color_continuous_scale=custom_scale,
        template='plotly_white'
    )
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title=metric,
        font=dict(size=13),
        height=500,
        margin=dict(l=60, r=30, t=60, b=60)
    )
    return pn.pane.Plotly(fig, config={'responsive': True}, sizing_mode='stretch_width')

# Layout
yearly_ratings_dashboard = pn.Column(
    "# Yearly Ratings Analysis",
    "## Click below to explore different metrics by year:",
    y_selector,
    pn.bind(plot_by_metric, y_selector)
)

#dashboard.show()


# %% [markdown]
# The first chart shows user activity rising significantly from 2003, peaking in 2006 with 34,380 ratings, then gradually declining to 5,282 ratings by 2015. This rise correlates with the emergence of recommender systems and the adoption of online movie ratings in the mid-2000s. The sharp decline after 2010 may indicate dataset limitations.
# 
# The second chart of average ratings per year is stable, with most years in a narrow range (3.45–3.6), indicating consistent behaviour. Exceptions are 1997, with a much higher average (4.34) due to a few early ratings skewing the result, and recent years (2014–2015) showing slight increases likely from fewer ratings amplifying variance.
# 
# Together, these visualizations are highlighting how both user volume and rating dynamics fluctuate over time. The trends may also inform content strategy, such as identifying high-engagement periods or understanding how audience behavior has evolved.

# %% [markdown]
# -------------------------------------------------------------

# %% [markdown]
# ## Tags Dataset Introduction
# 
# 
# The tags.csv dataset contains 94,875 user-generated tags applied to movies, along with corresponding user IDs and Unix timestamps. Each record represents a unique instance where a user assigned a free-text label to a specific movie. This kind of data provides qualitative insight into how users describe, interpret, and categorize movie content.
# 
# The presence of the 'timestamp' column also introduces a temporal component, allowing the study of tag usage over time, useful for tracking evolving trends, themes, or cultural relevance. Additionally, tags can be grouped by user or movie to examine tagging density, repetition, and topical diversity.
# 
# The dataset enhances movie profiling by combining structured genre data with user-defined tags, resulting in a richer semantic layer.

# %%
tags

# %%
tags.shape

# %%
tags.describe()

# %% [markdown]
# 
# 
# The dataset spans 94,875 tags added by users between approximately 2006 and 2015, based on Unix timestamps. The median timestamp corresponds to around 2007–2008, indicating peak tagging activity during the rise of online movie platforms. This temporal range provides useful context if tag trends are explored over time, even if the dataset is used in a supporting role.
# 

# %%
# Tag variable cleaning
tags['tag_clean'] = tags['tag'].str.lower().str.replace(r'[^\w\s]', '', regex=True).str.strip()


# %% [markdown]
# This code creates a new column tag_clean that stores a standardized version of each tag so that visually or semantically identical tags (likev 'Sci-Fi', 'sci fi', 'sci-fi') are treated the same.
# 
# This ensures that when we group or count tags later, we don't end up with duplicated categories caused by formatting differences.

# %%
# Convert Unix timestamp to datetime
tags['datetime'] = pd.to_datetime(tags['timestamp'], unit='s')

# %%
tags

# %%
# Tagging activity over time

tags['year'] = tags['datetime'].dt.year
tags['year'].value_counts().sort_index()

# %%
# Top users tag counts

tags['userId'].value_counts().head(10)


# %%
# Top tags
tags['tag_clean'].value_counts().head(15)


# %% [markdown]
# #### Top 15 Most Frequent Tags
# 
# This chart is created by extracting the 15 most frequent user-applied tags using `value_counts().head(15)` on the cleaned `tag_clean` column. The visualization has been built using `sns.barplot()`, with some enhancements:
# 
# - The color palette `sns.color_palette("OrRd_r", len(top_tags))` applies a warm gradient that improves visual engagement and guides the eye.
# - Font size and weight are manually adjusted in the title and axis labels to ensure clarity in presentation contexts.
# - The plot style `sns.set(style='whitegrid')` is applied to modernize the layout and improve data readability.
# 
# The visual style is intentionally distinct from previous EDA outputs to reflect a modern, dynamic approach—tailoring each dataset’s presentation to the nature of its content. In this case, the warm palette and aesthetic layout evoke a velvet-toned, cinematic feel, designed to resonate with the visual language of film culture and engage the project’s target audience.

# %%
# Prepare data
top_tags = tags['tag_clean'].value_counts().head(15)

# Set aesthetic style
sns.set(style='whitegrid')

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(
    x=top_tags.values,
    y=top_tags.index,
    palette=sns.color_palette("OrRd_r", len(top_tags))
)

# Titles and labels
plt.title('Top 15 Most Frequent Tags', fontsize=16, fontweight='bold')
plt.xlabel('Tag Count', fontsize=12)
plt.ylabel('Tag', fontsize=12)

# Font and spacing tweaks
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.tight_layout()
plt.show()


# %% [markdown]
# The bar chart showcases the 15 most frequently applied tags, revealing that users often describe films based on style (*atmospheric*, *quirky*), narrative (*twist ending*, *dystopia*), or tone (*dark comedy*, *surreal*) rather than just traditional genres. The prominence of *based on a book* suggests strong user interest in source material, while tags like *nudity topless* and *superhero* reflect both niche and mainstream content preferences.
# 
# The visual style using a warm gradient, intentionally evokes a cinematic, velvet-inspired tone. This aligns with the project’s design philosophy.

# %% [markdown]
# --------------------------------------------------

# %% [markdown]
# # Merging 'ratings' and 'movies'

# %%
movies1

# %%
ratings1

# %%
# Rename for clarity
ratings1 = ratings1.rename(columns={'year': 'rating_year'})
movies1 = movies1.rename(columns={'year': 'release_year'})


# %%
# Merge into final dataset
df = ratings1.merge(movies1, on='movieId', how='left')

# %%
df_genre = df.copy()

# %%
df_genre

# %%
df.columns.tolist()

# %% [markdown]
# # Merged Dataset EDA

# %% [markdown]
# ### Genre Rating Trends 
# 
# An interactive dashboard has been created to explore how average user ratings change across movie genres and release years. This dashboard is defined to uncover patterns in user ratings and genre evolution over time. This visualization coding development has gone through two interactive stages of improvement.
# 
# 
# The initial version allowed users to filter by genre using a `ToggleGroup`, and to define a custom release year range with an `IntRangeSlider`. While the structure was functional, several issues were identified during testing. The year slider label was too small and visually weak, the widget layout caused components to be truncated or misaligned on certain screen sizes, and the overall user interface lacked visual clarity.
# 
# To resolve these limitations, a second version of the dashboard has been implemented with a series of enhancements. The layout has been restructured to improve responsiveness, the widget label styling has been fully customised, and the entire dashboard has been adjusted for better readability and usability.
# 
# 
# 
# - `pn.pane.Markdown` has been used to create a clear and visually prominent label above the slider, replacing the default small and hard-to-read slider label.
# 
# - The `IntRangeSlider` has been set with `sizing_mode='stretch_width'` to ensure it scales appropriately and remains fully visible within the dashboard layout.
# 
# - The original layout used a single `pn.Row` for both widgets, which has now been replaced with a `pn.Column` structure to vertically stack the toggle group and slider, preventing any overlap or truncation.
# 
# - The plotting logic has remained consistent. The `plot_genre_trends()` function:
#   - filters the melted DataFrame based on selected genres and release year range,
#   - groups the results by `['Genre', 'release_year']`,
#   - calculates the mean `rating` for each group,
#   - and displays the result as a `Plotly Express` line chart with interactive markers.
# 
# - The layout includes `pn.bind()` to dynamically update the plot when widget values change. A fallback message is shown when no data matches the selected filters.
# 
# 
# 
# In addition to improving functionality, the visual design of the dashboard has also been refined. Custom HTML has been used in the title and subtitle markdown blocks to apply color, font size, and hierarchy, creating a more engaging and cinematic feel that aligns with the project’s domain.
# 
# This final version has addressed the key limitations observed in the original prototype and has resulted in a more accessible, maintainable, and professional interactive interface. It offers the user a clear view of how audience ratings evolve across genres and time, while also being presentation-ready and suitable for integration into a broader data storytelling dashboard.
# 

# %%
# Initializes Panel with Plotly backend enabled. Required to render Plotly figures inside Panel dashboards.
pn.extension('plotly')

# Step 1: Transform genre columns into rows so each row reflects a single genre label per movie
df_melted = df_genre.melt(
    id_vars=['rating', 'release_year'],  # Keep rating and release year as fixed columns
    value_vars=['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary',
                'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery',
                'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'],  # These are the genre flags
    var_name='Genre',  # New column for genre label
    value_name='IsGenre'  # New column for genre presence (1 or 0)
)
# Keep only the rows where a genre is present (IsGenre == 1)
df_melted = df_melted[df_melted['IsGenre'] == 1].copy()

# Step 2: Create interactive widgets to filter the dashboard
genres = sorted(df_melted['Genre'].unique())  # Get all genre names in alphabetical order

# Set up a toggle group to select multiple genres with visible styling
genre_selector = pn.widgets.ToggleGroup(
    name='Genres',
    value=['Drama'],  # Default selection
    options=genres,
    behavior='check',  # Allows multi-selection
    button_type='success'  # Green color to highlight selected genres
)

# Create a year range slider that adapts to the actual data range
year_slider = pn.widgets.IntRangeSlider(
    name='Select Release Year Range',
    start=df_melted['release_year'].min(),
    end=df_melted['release_year'].max(),
    value=(df_melted['release_year'].min(), df_melted['release_year'].max()),
    width=500
)

# Step 3: Define the function to generate the plot based on selected filters
def plot_genre_trends(genres, year_range):
    # Filter the melted DataFrame based on selected genres and year range
    filtered = df_melted[
        (df_melted['Genre'].isin(genres)) &
        (df_melted['release_year'].between(year_range[0], year_range[1]))
    ]
    
# Show a message if the filtered result is empty
    if filtered.empty:
        return pn.pane.Markdown("**No data available for the selected filters.**")

# Group by genre and year, then calculate the average rating for each combination
    trend_data = (
        filtered.groupby(['Genre', 'release_year'])['rating']
        .mean().reset_index()
    )

# Create the line plot using Plotly Express
    fig = px.line(
        trend_data,
        x='release_year',
        y='rating',
        color='Genre',
        markers=True,
        template='plotly_white',
        title='Average Rating by Genre Over Time',
        labels={'rating': 'Average Rating', 'release_year': 'Release Year'},
    )

# Improve layout: set height, font size, margin, and consistent hover behavior
    fig.update_layout(
        height=550,
        font=dict(size=13),
        xaxis=dict(tickmode='linear'),
        margin=dict(l=40, r=20, t=50, b=60),
        hovermode='x unified',
    )

# Return the Plotly figure embedded as a responsive Panel object
    return pn.pane.Plotly(fig, config={'responsive': True}, sizing_mode='stretch_width')

# Step 4: Assemble the final dashboard layout
'''dashboard = pn.Column(
    "# Genre Rating Trends Dashboard",  # Title
    "Click one or more genres and select a release year range to compare how ratings evolve over time:",  # Subtitle
    pn.Row(genre_selector, year_slider),  # Place widgets side by side
    pn.bind(plot_genre_trends, genre_selector, year_slider)  # Bind widgets to the plotting function
)

# Make the dashboard available to view
#dashboard.show()'''

# %%
# Initializes Panel with Plotly backend enabled. Required to render Plotly figures inside Panel dashboards.
pn.extension('plotly')
# Define a styled markdown label to display above the slider.
# This improves visual clarity compared to the default slider label.
year_slider_label = pn.pane.Markdown(
    "<h3 style='color:black; font-size:18px;'>Select Release Year Range:</h3>",
    sizing_mode='stretch_width'
)

# Create the interactive year range slider.
# The label is hidden by setting name to an empty string, and the width is set to stretch to avoid truncation.
year_slider = pn.widgets.IntRangeSlider(
    name='',
    start=int(df_melted['release_year'].min()),
    end=int(df_melted['release_year'].max()),
    value=(2000, 2015),
    step=1,
    sizing_mode='stretch_width'
)

# Disables display of the empty label field to maintain a clean layout.
year_slider.param.name.constant = False

# Define the plotting function that will respond to user input.
def plot_genre_trends(genres, year_range):
    # Filter the dataset by the selected genres and year range.
    filtered = df_melted[
        (df_melted['Genre'].isin(genres)) &
        (df_melted['release_year'].between(year_range[0], year_range[1]))
    ]

    # Display a message if the filtered dataset contains no results.
    if filtered.empty:
        return pn.pane.Markdown("**No data available for the selected filters.**")

    # Group the filtered data by genre and year, then calculate the average rating.
    trend_data = (
        filtered.groupby(['Genre', 'release_year'])['rating']
        .mean().reset_index()
    )

    # Create a line chart showing average ratings per genre over time.
    fig = px.line(
        trend_data,
        x='release_year',
        y='rating',
        color='Genre',
        markers=True,
        template='plotly_white',
        title='Average Rating by Genre Over Time',
        labels={'rating': 'Average Rating', 'release_year': 'Release Year'}
    )

    # Adjust layout and styling for clarity and responsiveness.
    fig.update_layout(
        height=550,
        font=dict(size=13),
        xaxis=dict(tickmode='linear'),
        margin=dict(l=40, r=20, t=50, b=60),
        hovermode='x unified'
    )

    # Return the plot for rendering within the dashboard.
    return pn.pane.Plotly(fig, config={'responsive': True}, sizing_mode='stretch_width')

# Construct the complete dashboard layout.
genre_trends_dashboard = pn.Column(
    # Main title of the dashboard.
    pn.pane.Markdown("<h1 style='color:#3E64FF; font-size:35px;'> Genre Rating Trends</h1>"),

    # Subtitle providing instructions for interaction.
    pn.pane.Markdown(
        "<p style='color:black; font-size:18px;'>Click one or more genres and select a release year range to compare how ratings evolve over time:</p>"
    ),

    # Interactive genre selector (ToggleGroup).
    genre_selector,

    # Display the styled slider label and the year slider widget.
    year_slider_label,
    year_slider,

    # Bind the plot function to the selected widgets.
    pn.bind(plot_genre_trends, genre_selector, year_slider),

    # Ensure the full layout stretches properly across the interface.
    sizing_mode='stretch_width'
)

# Make the dashboard available for display or serving.
#dashboard.show()


# %% [markdown]
# ### Genre Rating Trends Among 18–35-Year-Old Viewers (figure attached in report)
# 
# The filtered dashboard focuses on a 25-year release period (1990–2014), targeting movies relevant to the 18–35-year-old demographic. The genres selected (*Action, Adventure, Comedy, Fantasy, Horror, Romance, and Sci-Fi*) represent those most popular among younger adult audiences, based on recent entertainment surveys and streaming platform statistics (e.g., Statista, 2023; Morning Consult, 2022).
# 
# Across the examined genres, a gradual upward trend in average user ratings is observed over time, particularly after the year 2000. This pattern may reflect improvements in cinematic production, greater accessibility via digital platforms, and evolving audience preferences toward more immersive narratives.
# 
# #### Genre-Specific Analysis (1990–2014)
# 
# A closer examination of the selected genres reveals distinct patterns in user rating behaviour over the last 25 years. Notably, **Fantasy** and **Science Fiction** exhibit a marked upward trend in average ratings, particularly from the early 2000s onward. This increase aligns with the widespread popularity of cinematic universes and high-budget franchises such as *Harry Potter*, *Marvel*, and the *Star Wars* prequel and sequel trilogies. These genres have benefited from technological advancements, storytelling depth, and broad cultural appeal, resonating strongly with the 18–35-year-old demographic.
# 
# **Comedy** and **Romance**, on the other hand, display relatively stable average ratings throughout the period, typically ranging between 3.4 and 3.7. While these genres remain consistently present in popular media, their rating patterns suggest less dynamic growth.
# 
# The **Horror** genre is characterized by higher volatility in user ratings, reflecting its traditionally divisive reception. Despite this fluctuation, there is evidence of an upward trend after 2005, which may correspond to the rise of critically acclaimed psychological and socially reflective horror films.
# 
# Lastly, **Action** and **Adventure** maintain consistent audience engagement, with average ratings gradually increasing toward the end of the period. The genre’s appeal is likely driven by the success of global franchises, special effects innovation, and storytelling formats that favor recurring characters and episodic content—factors that align well with digital-native viewing habits.
# 
# Overall, these findings highlight evolving viewer preferences and suggest that genres delivering immersive experiences, visual innovation, and franchise continuity are gaining traction among younger audiences.
# 
# These patterns suggest that the 18–35 audience increasingly favors genres with rich world-building, emotional stakes, and visual innovation. The observed rating improvements support the idea that content tailored to digital-native preferences like serialized narratives, cinematic universes, and character depth, is being positively received. This dashboard confirms that genre preferences and satisfaction metrics are evolving in tandem with industry trends and audience expectations.
# 

# %% [markdown]
# ---

# %% [markdown]
# # K-Means Clustering

# %%
df

# %%
df1 = df.copy()

# %% [markdown]
# ### Dataset Filtering: Temporal and Genre-Based Refinement
# 
# To ensure the clustering analysis remains aligned with the objectives of this study and to understand patterns in movie preferences for younger audiences, the dataset has been filtered and refined as follows:
# 
# Although the `release_year` column indicates when a film was originally produced, the `rating_year` variable offers a more meaningful measure of user engagement — reflecting the specific moment when a viewer interacted with and rated a particular movie. Since this project focuses on understanding the preferences and behaviors of individuals aged 18 to 35, it is more appropriate to examine ratings from 1990 onwards.
# 
# By filtering based on `rating_year ≥ 1990`, the dataset retains not only contemporary films but also older titles that continue to be watched and valued by modern audiences. This approach ensures the inclusion of culturally significant classics that remain relevant within the digital streaming era. Moreover, it mitigates potential biases introduced by outdated rating patterns or sparsely rated films from earlier decades. As a result, the dataset becomes more representative of current audience behavior, particularly within the age demographic that drives the majority of digital engagement today.
# 

# %%
df1 = df1[df1['rating_year'] >= 1990].copy()

# %%
df1

# %% [markdown]
# In the original dataset, genre flags are encoded as binary indicators to capture each movie’s thematic composition. While this representation is effective for content-based analysis, not all genre categories contribute equally to clustering models. Certain genres exhibit very low frequency across the dataset, while others represent technical attributes rather than narrative style.
# 
# Genres such as *Film-Noir*, *Documentary*, *Western*, *Musical*, and *War* appear only sporadically and are either outdated or highly specific in style. Additionally, *IMAX* reflects a projection format rather than a thematic genre, making it unsuitable for semantic clustering. Including such variables could distort similarity calculations by introducing sparsity or irrelevant distinctions into the feature space.
# By removing these underrepresented or non-thematic genres, the dimensionality of the dataset is reduced without sacrificing meaningful variance.
# 
# 

# %%
# Get all genre columns to list and then sum the count of every genre
genre_cols = df1.columns[df1.dtypes == 'int64'].tolist()
genre_usage = df1[genre_cols].sum().sort_values()

# View least-used genres
print(genre_usage.head(10))


# %%
df1 = df1.drop(columns=['Film-Noir', 'Documentary', 'Western', 'Musical', 'War', 'IMAX'])

# %%
df1

# %% [markdown]
# ### Clustering Feature Selection
# 
# The dataset is refined to include only features that describe each movie’s content and audience reception. Columns such as `userId`, `timestamp`, `rating`, `title`, and calendar-related variables are removed, as they either lack analytical value or duplicate information already aggregated.
# 
# The final dataset retains variables like `avg_rating_movie`, `num_ratings_movie`, `rating_std_movie`, and `rating_year`, alongside one-hot encoded genre flags. These features allow the clustering models to group movies based on both their thematic structure and how audiences respond to them.
# 

# %%
# Drop columns
df_cluster = df1.drop(columns=['userId', 'timestamp', 'rating', 'title', 'month', 'dayofweek', 'userId'])


# %%
# # Aggregate rating statistics per movie and merge with genre flags.
df_cluster = (
    df1.groupby(['movieId', 'release_year', 'rating_year'])
       .agg(avg_rating_movie=('rating', 'mean'),
            num_ratings_movie=('rating', 'count'),
            rating_std_movie=('rating', 'std'))
       .reset_index()
       .merge(df1.drop_duplicates('movieId')[[col for col in df1.columns if col in genre_cols + ['movieId']]],
              on='movieId', how='left')
       .dropna()
)


# %%
df_cluster

# %% [markdown]
# ### K-Means Clustering Preparation with PCA
# 
# K-Means clustering has been implemented to identify natural groupings in the movie dataset based on numeric characteristics such as average rating, rating variability, and genre indicators.
# 
# Before applying clustering, all features have been scaled using `StandardScaler()` to ensure uniform influence in Euclidean space and dimensionality has been reduced using `PCA(n_components=2)` to allow for visual inspection of clusters in two dimensions.
# 
# The K-Means model has been initialized with specific non-default parameters:
# - `n_clusters=3` to define the number of target groupings based on prior exploratory assumptions.
# - `max_iter=300` ensures sufficient iterations for convergence.
# - `n_init=10` runs the algorithm multiple times with different centroid seeds to select the optimal result.
# - `random_state=38` has been set for reproducibility.
# 

# %%
# Select features for clustering
features = df_cluster.drop(columns=['movieId', 'release_year', 'rating_year']).columns
X = df_cluster[features]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %%
# Define KMeans with k=3
kmeans = KMeans(n_clusters=3, max_iter=300, n_init=10, random_state=38)

# Fit model and predict clusters
y_kmeans = kmeans.fit_predict(X_scaled)

# Reducing dimensions with PCA to visualize clusters in 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
centroids_pca = pca.transform(kmeans.cluster_centers_)

# Visualize clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[y_kmeans == 0, 0], X_pca[y_kmeans == 0, 1], s=50, c='red', label='Cluster 1')
plt.scatter(X_pca[y_kmeans == 1, 0], X_pca[y_kmeans == 1, 1], s=50, c='blue', label='Cluster 2')
plt.scatter(X_pca[y_kmeans == 2, 0], X_pca[y_kmeans == 2, 1], s=50, c='green', label='Cluster 3')

# Plot centroids
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=80, c='yellow', label='Centroids', marker='X')

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('K-Means Clustering of Movies')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()


# %% [markdown]
# The plot above illustrates the distribution of movies across three clusters based on their scaled feature values. PCA has been applied to reduce the data to two principal components, enabling clear visual separation of groupings. While clusters show some level of distinction, moderate overlap remains, supporting the earlier silhouette score interpretation that further validation (e.g., elbow method) may be beneficial.
# 

# %%
# Display the silhouette score values
print(f'Silhouette Score(n = 3): {silhouette_score(X_scaled, y_kmeans)}')

# %% [markdown]
# To evaluate the quality of the K-Means clustering with `n=3`, the silhouette score has been computed. This metric quantifies how well each point fits within its assigned cluster compared to others. The result is showing a score of 0.15 suggesting a modest but present clustering structure. Some degree of separation is achieved, though overlap remains. While this is a reasonable starting point, it may not represent the optimal number of clusters.

# %% [markdown]
# ### Elbow Method for Optimal Cluster Selection
# 
# To determine the appropriate number of clusters for K-Means, the Elbow Method has been applied. This technique plots the Within-Cluster Sum of Squares (WCSS) for a range of cluster values. The point where the WCSS curve begins to flatten ("elbow") suggests a suitable number of clusters that balances compactness and simplicity.

# %%
#Elbow method

wcss = []

# Loop through cluster sizes from 2 to 10
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, max_iter=300, n_init=10, random_state=38)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), wcss, marker='o')
plt.title('Elbow Method to Determine Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Inertia)')
plt.grid(True)
plt.tight_layout()
plt.show()


# %% [markdown]
# ### K-Means Cluster Evaluation
# 
# To evaluate the optimal number of clusters (`k`) for K-Means, two metrics are plotted:
# 
# - **WCSS (Inertia):** Measures cluster compactness; lower is better.
# - **Silhouette Score:** Measures separation between clusters; higher is better.
# 
# A `for` loop iterates from `k=2` to `k=8`, storing both metrics for each model. The resulting plot uses dual y-axes to compare cluster cohesion and separation simultaneously, supporting informed selection of `k`.

# %%

# Range of cluster values to test
k_range = range(2, 8)

# Storage for WCSS and silhouette scores
wcss = []
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, max_iter=300, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

# Plotting WCSS and Silhouette Score
fig, ax1 = plt.subplots(figsize=(9, 6))

color = 'tab:blue'
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('WCSS (Inertia)', color=color)
ax1.plot(k_range, wcss, marker='o', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_title('K-Means Evaluation: WCSS vs Silhouette Score')

ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('Silhouette Score', color=color)
ax2.plot(k_range, silhouette_scores, marker='o', linestyle='--', color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.tight_layout()
plt.grid(True)
plt.show()


# %% [markdown]
# ### Interpretation of Clustering Evaluation
# 
# Based on the graph, `k = 2` emerges as the most suitable number of clusters. While WCSS continues to decrease as expected, the Silhouette Score peaks at `k = 2`, suggesting the highest balance between compactness and separation. Higher values of `k` show diminishing gains or instability in silhouette performance, reinforcing `k = 2` as the optimal choice for the next clustering step.
# 

# %%
# Apply KMeans with k=5 on already scaled data
kmeans_2 = KMeans(n_clusters=2, max_iter=300, n_init=10, random_state=42)
y_kmeans_2 = kmeans_2.fit_predict(X_scaled)

# Use existing PCA projection (X_pca) from earlier
centroids_pca_2 = pca.transform(kmeans_2.cluster_centers_)

# Plot
plt.figure(figsize=(9, 7))
colors = ['red', 'blue', 'green', 'purple', 'orange']
for i in range(5):
    plt.scatter(X_pca[y_kmeans_2 == i, 0], X_pca[y_kmeans_2 == i, 1],
                s=40, c=colors[i], label=f'Cluster {i+1}')
    
plt.scatter(centroids_pca_2[:, 0], centroids_pca_2[:, 1],
            s=100, c='yellow', edgecolors='black', marker='X', label='Centroids')

plt.title('K-Means Clustering of Movies (k = 2)', fontsize=14)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()


# %% [markdown]
# ### K-Means Clustering Results (k = 2)
# 
# A two-cluster solution has been applied based on the combination of inertia and silhouette score evaluations. While the PCA visualization shows some degree of separation between clusters, this visual dispersion aligns with the relatively low silhouette coefficient (≈ 0.25), suggesting that the clusters may not be well-separated in high-dimensional space.
# 
# These results indicate that K-Means may not be the most suitable technique for this dataset’s structure, potentially due to the linear cluster assumptions and variance sensitivity.

# %% [markdown]
# ## Alternative Clustering Dataset
# 
# 
# A refined dataset has been created to improve clustering performance and interpretability by aggregating ratings at the movie level. This removes repeated user-level rows and emphasizes movie characteristics rather than individual behaviors.
# 
# The dataset includes the average rating, total number of ratings, and rating variability per movie, along with binary genre indicators (e.g., Action, Comedy, Sci-Fi).
# 
#  Two new temporal features have been added: `rating_year_range`, which captures the span between the first and last rating year as a measure of cultural longevity; and `rating_trend`, which reflects whether a movie’s popularity increased or decreased over time based on a fitted linear regression.

# %%
# Define the genre columns to retain for clustering
genre_cols = [col for col in df_cluster.columns if col in [
    'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
    'Drama', 'Fantasy', 'Horror', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller'
]]

# %%
# Compute the rating span per movie
rating_year_range = (
    df_cluster.groupby('movieId')['rating_year']
    .agg(['min', 'max'])
    .assign(rating_year_range=lambda x: x['max'] - x['min'])
    .drop(columns=['min', 'max'])
    .reset_index()
)

# Compute the rating trend (slope of ratings over years)
from sklearn.linear_model import LinearRegression

rating_trends = []
for movie_id, group in df_cluster.groupby('movieId'):
    group_sorted = group.sort_values('rating_year')
    X = group_sorted[['rating_year']].values
    y = group_sorted['avg_rating_movie'].values
    if len(X) > 1 and len(np.unique(X)) > 1:
        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0]
    else:
        slope = np.nan
    rating_trends.append({'movieId': movie_id, 'rating_trend': slope})

df_trend = pd.DataFrame(rating_trends)


# %%
# Aggregate numeric statistics per movie
df_movies = (
    df_cluster.groupby(['movieId', 'release_year'], as_index=False)
    .agg(
        avg_rating_movie=('avg_rating_movie', 'mean'),
        num_ratings_movie=('num_ratings_movie', 'sum'),
        rating_std_movie=('rating_std_movie', 'mean')
    )
)


# %%
# Attach genre flags (deduplicated by movieId)
genre_data = df_cluster.drop_duplicates('movieId')[['movieId'] + genre_cols]
genre_data = genre_data.loc[:, ~genre_data.columns.duplicated()]  # Just in case

# Merge all auxiliary features into final dataset
df_movies = (
    df_movies
    .merge(genre_data, on='movieId', how='left')
    .merge(rating_year_range, on='movieId', how='left')
    .merge(df_trend, on='movieId', how='left')
)


# %% [markdown]
# To improve clustering reliability, outliers are removed using Z-scores on key numeric features: `avg_rating_movie`, `num_ratings_movie`, `rating_std_movie`, `rating_year_range`, and `rating_trend`. Values exceeding 3 standard deviations are excluded to eliminate extreme cases and reduce noise in the dataset.
# 

# %%
# Select only continuous numerical features for outlier detection
num_features = ['avg_rating_movie', 'num_ratings_movie', 'rating_std_movie', 'rating_year_range', 'rating_trend']

# Drop NaNs (important for rating_trend)
df_clean = df_movies.dropna(subset=num_features).copy()

# Calculate Z-scores
z_scores = np.abs(zscore(df_clean[num_features]))

# Define threshold and filter rows with any Z-score > 3
outliers = (z_scores > 3).any(axis=1)
print(f"Outliers detected: {outliers.sum()} of {len(df_clean)}")

# Filter out outliers
df_movies_clean = df_clean[~outliers].reset_index(drop=True)


# %%
df_movies_clean

# %%
df_movies_clean = df_movies_clean.loc[:, ~df_movies_clean.columns.duplicated()]

# %% [markdown]
# ### Interactive K-Means Clustering Dashboard
# 
# This section implements a flexible and modular clustering tool to test various groupings of movies based on selected numeric features. The panel includes:
# 
# - Dynamic feature selection: `x` and `y` axes using `pn.widgets.Select`, enabling on-the-fly testing of different variable combinations.
# - Cluster control: `IntSlider` to adjust `k`, supporting exploration from 2 to 8 groups.
# - StandardScaler(): Applied to ensure input normalization across different feature scales—critical for distance-based clustering.
# - Silhouette Score Calculation: Embedded directly in the figure title (`plotly.express`) using `silhouette_score(X_scaled, labels)` for real-time quality assessment of cluster separation.
# - Centroid visualization: Reverse-scaled and displayed using custom markers (`symbol='x'`), aiding visual interpretability.
# 
# 
# Rather than fixing the analysis on a single pair of dimensions, this setup allows testing various hypotheses about what might drive meaningful segmentation (e.g., popularity vs. consistency, engagement vs. trend). Including the silhouette score gives immediate quantitative feedback on how distinct the resulting clusters are for each configuration.
# 
# 

# %%
pn.extension('plotly')

# Select numerical features
numeric_features = [
    'avg_rating_movie', 'num_ratings_movie', 'rating_std_movie',
    'rating_year_range', 'rating_trend'
]

# Widgets for interactive selection
x_axis = pn.widgets.Select(name='X-Axis Feature', options=numeric_features, value='avg_rating_movie')
y_axis = pn.widgets.Select(name='Y-Axis Feature', options=numeric_features, value='rating_trend')
k_slider = pn.widgets.IntSlider(name='Number of Clusters (k)', start=2, end=8, value=3)

# Interactive plotting function
def interactive_cluster_plot(x_col, y_col, k):
    df_clean = df_movies_clean[[x_col, y_col]].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    silhouette = silhouette_score(X_scaled, labels)

    df_plot = df_clean.copy()
    df_plot['Cluster'] = labels
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)

    fig = px.scatter(
        df_plot, x=x_col, y=y_col, color=df_plot['Cluster'].astype(str),
        title=f"K-Means Clustering (k={k})<br>Silhouette Score: {silhouette:.3f}",
        height=600
    )

    fig.add_scatter(
        x=centroids[:, 0], y=centroids[:, 1],
        mode='markers', marker=dict(size=14, color='black', symbol='x'),
        name='Centroids'
    )

    return pn.pane.Plotly(fig, config={'responsive': True}, sizing_mode='stretch_width')

# Bind interactive output
interactive_panel = pn.bind(interactive_cluster_plot, x_col=x_axis, y_col=y_axis, k=k_slider)

# Build dashboard layout
kmeans_dashboard = pn.Column(
    pn.pane.Markdown("## Interactive K-Means Clustering Dashboard"),
    pn.Row(pn.WidgetBox(x_axis, y_axis, k_slider, width=300), interactive_panel)
)

# Launch app in browser window
#dashboard.show()

# %% [markdown]
# Two feature combinations have shown the most meaningful segmentation performance in the interactive clustering dashboard. When plotting `num_ratings_movie` against `avg_rating_movie` with k = 3, the model achieves a Silhouette Score of 0.437. This configuration highlights a clear separation between niche films with low ratings, moderately rated movies with fewer reviews, and widely reviewed, high-rated titles. It effectively captures both user engagement and perceived quality.
# 
# Similarly, using `avg_rating_movie` versus `rating_std_movie` with k = 4 results in a Silhouette Score of 0.347. This pairing allows the audience to observe how polarizing a film is, based on the variance in user opinions. Clusters reflect consistent appreciation, high disagreement, or uniformly low-rated content. The inclusion of rating dispersion adds nuance to how viewer consensus is interpreted in the clustering process.
# 
# These plots have been selected not only for their statistical performance but also for their thematic clarity, supporting meaningful differentiation across audience engagement and sentiment.

# %% [markdown]
# # K-Medoids Clustering

# %% [markdown]
# This dashboard replicates the dynamic setup of the previous clustering tool, but now leverages the **K-Medoids** algorithm. Unlike K-Means, which assigns cluster centers based on mean values, K-Medoids selects actual data points as medoids, offering greater robustness to outliers and asymmetric distributions.
# 
# - `KMedoids(n_clusters=k, random_state=42)`: Used in place of KMeans.
# - Cluster centers representing real movie entries rather than calculated averages.
# - The silhouette score is updated live for every selection of features and number of clusters.
# - Plotly's scatter plot is used with labeled medoids marked by black X’s for clear reference.
# 
# To ensure visual clarity, medoid centroids are plotted last using `go.Scatter`, ensuring they appear clearly on top of the data points. Additionally, the cluster points have been rendered semi-transparent (`opacity=0.5`) to reduce visual clutter and make the medoid markers more distinguishable.
# 
# This tool enhances interpretability by allowing flexible experimentation with numeric features such as `rating_std_movie`, `rating_year_range`, and `rating_trend`, offering new perspectives on how different subsets of movies naturally group together.
# 
# 

# %%
def interactive_cluster_plot(x_col, y_col, k):
    df_clean = df_movies_clean[[x_col, y_col]].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)

    kmedoids = KMedoids(n_clusters=k, random_state=42)
    labels = kmedoids.fit_predict(X_scaled)
    silhouette = silhouette_score(X_scaled, labels)

    df_plot = df_clean.copy()
    df_plot['Cluster'] = labels
    centroids = scaler.inverse_transform(kmedoids.cluster_centers_)

    fig = px.scatter(
    df_plot, x=x_col, y=y_col, color=df_plot['Cluster'].astype(str),
    title=f"K-Medoids Clustering (k={k})<br>Silhouette Score: {silhouette:.3f}",
    height=600
)

# Adjust cluster points with semi-transparency
    fig.update_traces(marker=dict(size=7, opacity=0.5), selector=dict(mode='markers'))

# Add medoids on top
    fig.add_trace(go.Scatter(
    x=centroids[:, 0], y=centroids[:, 1],
    mode='markers',
    marker=dict(size=16, color='black', symbol='x'),
    name='Medoids',
    showlegend=True
))


    return pn.pane.Plotly(fig, config={'responsive': True}, sizing_mode='stretch_width')



# Bind and layout
interactive_panel = pn.bind(interactive_cluster_plot, x_col=x_axis, y_col=y_axis, k=k_slider)

kmedoids_dashboard = pn.Column(
    pn.pane.Markdown("## Interactive K-Medoids Clustering Dashboard"),
    pn.Row(pn.WidgetBox(x_axis, y_axis, k_slider, width=300), interactive_panel)
)

#dashboard.show()



# %% [markdown]
# K-Medoids produces meaningful results with two feature combinations. Plotting `num_ratings_movie` against `avg_rating_movie` (k = 3) have produced a Silhouette Score of 0.417, identifying clusters of popular high-rated titles, low-rated niche films, and a middle group with moderate engagement. Using real movies as medoids improved clarity in interpreting each segment.
# 
# In the case of `avg_rating_movie` versus `rating_std_movie` (k = 2), the Silhouette Score reached 0.399, revealing a division between consistently rated films and those with greater disagreement among viewers.
# 
# While performance is comparable to K-Means, K-Medoids offered the advantage of using actual movies as cluster centers, which may support clearer interpretation in practical settings. The reason is that K-Medoids is theoretically more robust to outliers because it uses real data points instead of means, which are sensitive to extreme values.

# %% [markdown]
# # Fuzzy C-Means Clustering

# %% [markdown]
# To prepare the dataset for fuzzy clustering, genre information has been merged into a single `genre_combo` column using `apply()` and string joining logic. A new numerical feature `genre_count` has been derived by summing across the original binary genre columns, reflecting the number of active genres per movie.
# 
# The original individual genre columns are removed to reduce sparsity. A `release_decade` feature has been also added by flooring `release_year`, enabling temporal comparison across clusters. These transformations ensured a clean feature set for scaling and clustering in the subsequent step.

# %%
df_movies_clean1 = df_movies_clean.copy()

# %%
# Define your genre columns
genre_cols = [
    'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
    'Drama', 'Fantasy', 'Horror', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller'
]

# Combine active genres into a single string per row
df_movies_clean1['genre_combo'] = df_movies_clean1[genre_cols].apply(
    lambda row: ','.join([genre for genre in genre_cols if row[genre] == 1]),
    axis=1
)


# %%
df_movies_clean1['genre_count'] = df_movies_clean1[genre_cols].sum(axis=1)


# %%
df_movies_clean1 = df_movies_clean1.drop(columns=genre_cols)

# %%
# Adding a Release Decade feature
df_movies_clean1['release_decade'] = (df_movies_clean1['release_year'] // 10) * 10

# %%
df_movies_clean1

# %% [markdown]
# ### Cluster Evaluation
# 
# 
# A set of numerical features is selected for clustering, including rating metrics, genre complexity, and temporal indicators. These features are scaled using `StandardScaler()` before applying `FCM()` across different cluster values (`k = 2` to `6`).
# 
# To evaluate model quality, the Fuzzy Partition Coefficient (FPC) is computed using a custom `compute_fpc()` function. As `k` increases, FPC values decrease, indicating lower separation. The highest score is obtained at `k = 2` (FPC = 0.5000), suggesting that two clusters provide the clearest segmentation while maintaining fuzzy boundaries.
# 
# 

# %%
features = [
    'avg_rating_movie',
    'num_ratings_movie',
    'rating_std_movie',
    'rating_year_range',
    'rating_trend',
    'genre_count',
    'release_decade'
]

X = df_movies_clean1[features].to_numpy()


# %%
X = df_movies_clean1[features].dropna().values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# %%
# Store clustering results
fcm_results = {}

# Try different cluster counts
for k in range(2, 7):
    fcm = FCM(n_clusters=k, max_iter=1000)
    fcm.fit(X_scaled)
    
    labels = fcm.u.argmax(axis=1)
    membership = (fcm.u * 100).astype(int)
    centers = fcm.centers
    
    # Save all outputs for this k
    fcm_results[k] = {
        'labels': labels,
        'membership_matrix': membership,
        'centers': centers,
        'model': fcm
    }

# %%
def compute_fpc(u_matrix):
    n_samples = u_matrix.shape[0]  # correct shape: (samples, clusters)
    return np.sum(u_matrix ** 2) / n_samples

fpc_scores = {}

for k, result in fcm_results.items():
    fcm_model = result['model']
    u_matrix = fcm_model.u  # shape: [n_clusters, n_samples]
    fpc = compute_fpc(u_matrix)
    fpc_scores[k] = fpc
    print(f"Clusters: {k} → FPC: {fpc:.4f}")



# %% [markdown]
# ### Tuning the Fuzziness Coefficient `m`
# 
# To optimize the soft clustering behavior, the fuzziness coefficient `m` has been tuned using `FCM(n_clusters=2, m=...)`. This parameter controls the degree to which data points can belong to multiple clusters:
# 
# - Lower values of `m` (closer to 1.0) produce sharper, crisper cluster boundaries, similar to K-Means.
# - Higher values allow for more overlap, increasing flexibility in cases with ambiguous or mixed observations.
# 
# The tuning is performed for `m` values ranging from 1.5 to 2.5. For each configuration, the model has been fitted on the standardized feature space (`X_scaled`) and evaluated using the Fuzzy Partition Coefficient (FPC).

# %%
for m_val in [1.5, 1.7, 2.0, 2.3, 2.5]:
    fcm = FCM(n_clusters=2, m=m_val, max_iter=1000)
    fcm.fit(X_scaled)
    fpc = compute_fpc(fcm.u)
    print(f"FPC with m={m_val:.1f}: {fpc:.4f}")


# %%
fcm_best = FCM(n_clusters=2, m=1.5, max_iter=1000)
fcm_best.fit(X_scaled)

final_labels = fcm_best.u.argmax(axis=1)
df_movies_clean1['fuzzy_cluster'] = final_labels

# How strongly each movie belongs to its assigned cluster
df_movies_clean1['fuzzy_membership'] = fcm_best.u.max(axis=1)


# %%
# Filter medium-high confident movies
df_movies_clean1 = df_movies_clean1[df_movies_clean1['fuzzy_membership'] >= 0.7]
df_movies_clean1 

# %% [markdown]
# 
# The optimal configuration is obtained with `m = 1.5`, which yielded the highest FPC (0.5964). This suggests that the model benefits from sharper cluster boundaries in this context, as higher fuzziness levels led to uniform membership distributions and poorer discrimination.
# 
# Tuning 'm' enables better control over cluster interpretability. In datasets like this, where structural patterns exist but overlap is moderate, reducing fuzziness sharpens group identity without fully removing the flexibility of soft clustering. However, it is important to think that using a lower 'm' value sharpens the clusters, but it can oversimplify movies that naturally belong to multiple groups (e.g., hybrid-genre or polarizing titles).
# 
# The final clustering model has been fitted using `FCM(n_clusters=2, m=1.5)` based on the previously observed FPC improvements. Hard cluster labels are assigned by applying `argmax()` to the membership matrix, and stored in the `fuzzy_cluster` column.
# 
# To retain interpretability, a `fuzzy_membership` column has been added, which captures how strongly each movie belongs to its assigned cluster. This value provides insight into model confidence and is used to support fuzzy interpretation in subsequent analysis.

# %%
# Define columns to compare across clusters
compare_cols = [
    'avg_rating_movie',
    'num_ratings_movie',
    'rating_std_movie',
    'rating_trend',
    'genre_count',
    'rating_year_range',
    'release_decade'
]

# Group by cluster and summarize
cluster_summary = df_movies_clean1.groupby('fuzzy_cluster')[compare_cols].mean().round(2)

# Add counts per cluster
cluster_summary['count'] = df_movies_clean1['fuzzy_cluster'].value_counts().sort_index()

pd.set_option('display.max_columns', None)
cluster_summary

# %% [markdown]
# ### Cluster Interpretation
# 
# Following clustering, descriptive statistics have been computed for each segment. Cluster 0 is composed of older films (mean release 1970s decade) with higher average ratings (3.75), fewer genres per title, and longer rating lifespans. These characteristics suggest content with enduring appeal and more defined thematic focus.
# 
# Cluster 1, by contrast, contains newer movies (1990s) with broader genre combinations (`genre_count = 2.81`) and lower average ratings (2.93). A higher standard deviation (0.93) shows greater viewer disagreement, and a shorter rating lifespan indicates more temporary popularity.
# 
# These differences support a clear segmentation between consistent, well-received legacy titles and more complex, polarizing modern content. The results align with expectations for fuzzy clustering, where soft boundaries help capture such behavioral variation across temporal dimensions.
# 

# %%
pn.extension('plotly')

# Define a styled title and subtitle
title = pn.pane.Markdown(
    "<h1 style=font-size:35px;'>Fuzzy Cluster Explorer</h1>",
    sizing_mode='stretch_width'
)

subtitle = pn.pane.Markdown(
    "<p style='color:black; font-size:18px;'>Select two features to visualize the clustering structure of movies based on fuzzy membership strength.</p>",
    sizing_mode='stretch_width'
)

# Axis selectors with default features
axis_options = [
    'avg_rating_movie', 'num_ratings_movie', 'rating_std_movie',
    'rating_year_range', 'rating_trend', 'genre_count', 'release_decade'
]

x_selector = pn.widgets.Select(name='X-Axis', options=axis_options, value='avg_rating_movie')
y_selector = pn.widgets.Select(name='Y-Axis', options=axis_options, value='num_ratings_movie')

# Plotting function for interactive chart
def plot_fuzzy_clusters(x, y):
    fig = px.scatter(
        df_movies_clean1,
        x=x, y=y,
        color=df_movies_clean1['fuzzy_cluster'].astype(str),
        size='fuzzy_membership',
        hover_data=['genre_combo', 'avg_rating_movie', 'num_ratings_movie'],
        opacity=0.65,
        title=f'Fuzzy C-Means Clustering: {x} vs {y}',
        labels={'color': 'Cluster'},
        template='plotly_white'
    )

    fig.update_layout(
        height=550,
        font=dict(size=13),
        margin=dict(l=40, r=20, t=50, b=60),
        hovermode='closest'
    )

    return pn.pane.Plotly(fig, config={'responsive': True}, sizing_mode='stretch_width')

# Bind selectors to plotting function
fuzzy_dashboard = pn.Column(
    title,
    subtitle,
    pn.Row(x_selector, y_selector, sizing_mode='stretch_width'),
    pn.bind(plot_fuzzy_clusters, x_selector, y_selector),
    sizing_mode='stretch_width'
)

# Display dashboard
#dashboard.show()


# %% [markdown]
# ### Fuzzy Cluster Interactive Panel
# 
# 
# This dashboard is built using `Panel` and Plotly to enable flexible, real-time visual exploration of fuzzy clustering results. It includes two dropdown widgets for selecting the x- and y-axis features, making it easy to explore different variable combinations without restarting or modifying the code.
# 
# Each point’s size reflects its fuzzy membership value (`fuzzy_membership`), so more confidently assigned movies appear larger. Marker transparency is manually adjusted (`opacity=0.65`) to reduce visual clutter in high-density areas.
# 
# The plotting function is linked to the widgets using `pn.bind()`, allowing the chart to update instantly whenever a new feature is selected. The layout uses `pn.Column()` and `pn.Row()` with `sizing_mode='stretch_width'` to ensure responsive behavior across screen sizes. Unlike previous dashboards, this one runs entirely within the notebook and does not rely on a `servable()` deployment or external template.
# 
# This setup supports interpretability and interactive cluster validation while remaining simple and efficient to use.
# 
# 

# %% [markdown]
# ---

# %% [markdown]
# # User-User Collaborative Filtering

# %% [markdown]
# ### Feature Selection
# 
# The 'ratings1' dataset contains 264,505 rows with `userId`, `movieId`, and `rating` as key fields. Only these columns have been retained for collaborative filtering. This structure has ensured the dataset matched the required user-item-rating format.
# Collaborative filtering relies solely on interaction data. Therefore, contextual features such as timestamp, avg_rating_movie, or temporal metadata have been excluded to maintain a pure similarity-based model.

# %%
ratings1

# %%
# Prepare the dataset for Collaborative Filtering
df_cf = ratings1[['userId', 'movieId', 'rating']]

# %%
df_cf.head()

# %%
df_cf[df_cf['rating']>0].describe()

# %% [markdown]
# The rating distribution has shown a mean of 3.50, with full coverage of the 0.5–5.0 scale. This confirmed sufficient variability for similarity computation.

# %%
df_cf.set_index("userId", inplace=True)

# %% [markdown]
# ### Active Users
# 
# To reduce sparsity and improve similarity accuracy, users who had rated fewer than 200 movies have been excluded. This threshold ensures the model is built on consistently engaged users.
# 

# %%
active_users = df_cf.groupby("userId").count()["movieId"] > 200
active_user_ids = active_users[active_users].index
df_cf = df_cf[df_cf.index.isin(active_user_ids)]
active_user_ids

# %% [markdown]
# A total of 446 active users have been retained for the similarity matrix.
# The filtered ratings have been transformed into a user-item matrix with explicit ratings. Missing values have been filled with 0 to indicate no interaction:

# %%
user_item_matrix = df_cf.reset_index().pivot(index='userId', columns='movieId', values='rating').fillna(0)
df_cf

# %%
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)


# %%
user_similarity_df

# %% [markdown]
# ### User-User Similarity Matrix
# 
# This matrix captures the similarity scores between users based on their movie ratings. Each cell represents how similar user is to another user, using a similarity metric such as cosine similarity.
# 
# - Values range from **0 to 1**, where **1** indicates perfect similarity.
# - The matrix is **symmetric** with **1.0** on the diagonal (user compared to self).
# - Used in **user-based collaborative filtering** to find nearest neighbors and generate personalized movie recommendations.
# 
# 
# 
# This structure is helping in user-based collaborative filtering, where the most similar users (top-k neighbors) are identified for a given user. These neighbors' ratings can then be used to estimate how the user might rate unseen movies, often by computing a weighted average of the neighbors' ratings, with the similarity scores. Based on these predictions, personalized movie recommendations can be generated by selecting titles that highly similar users have rated positively.

# %%
user_similarity_df.shape

# %%
user_mean_rating = df_cf.reset_index().groupby('userId')['rating'].mean()
user_mean_rating = user_mean_rating.to_frame(name='mean_rating')
user_mean_rating

# %%
user_ratings_merged = pd.merge(df_cf, user_mean_rating, on="userId")

# %%
user_ratings_merged

# %%
user_ratings_merged["Adjusted-Rating"] = user_ratings_merged["rating"] - user_ratings_merged["mean_rating"]

# %%
user_ratings_merged

# %% [markdown]
# To normalize individual rating behaviors, the mean rating of each user is computed. This accounts for personal bias (some users tend to rate consistently higher or lower than others).
# 
# The original ratings are then adjusted by subtracting the corresponding user’s mean. This produces *adjusted ratings*, which center user preferences around zero, allowing for fairer comparison across users.
# 
# This transformation helps improve the effectiveness of similarity-based predictions by focusing on relative preferences rather than absolute scales.
# 

# %% [markdown]
# **Cosine Similarity Based on Adjusted Ratings**
# 
# A user-item matrix is created from adjusted ratings to center user preferences. This matrix is built using `pivot_table()` with `fill_value=0`.

# %%
user_item_adj_matrix = user_ratings_merged.reset_index().pivot_table(
    index='userId',
    columns='movieId',
    values='Adjusted-Rating',
    fill_value=0
)

user_item_adj_matrix


# %% [markdown]
# Cosine similarity is then computed with `cosine_similarity()`, and diagonal values have been set to zero to exclude self-similarity. The result, `similarity_with_user`, captures how similar users are in rating behavior, independent of overall rating scale.

# %%
b = cosine_similarity(user_item_adj_matrix)
np.fill_diagonal(b, 0)
similarity_with_user = pd.DataFrame(b, index=user_item_adj_matrix.index, columns=user_item_adj_matrix.index)
similarity_with_user.head()


# %% [markdown]
# ### Top Similar Users
# 
# Using the cosine similarity matrix, the `find_n_neighbours()` function identifies the top-N most similar users for each user by sorting similarity scores in descending order (excluding self-similarity). For example, `top_30_similar_users` captures the 30 nearest neighbors per user, enabling collaborative filtering based on peer preferences.

# %%
def find_n_neighbours(sim_df, n):
    top_n = sim_df.apply(
        lambda row: pd.Series(
            row.sort_values(ascending=False).iloc[:n].index,
            index=[f'top{i+1}' for i in range(n)]
        ),
        axis=1
    )
    return top_n

# %%
# Find top 30 most similar users for each user
top_30_similar_users = find_n_neighbours(similarity_with_user, 30)
top_30_similar_users.head()

# %% [markdown]
# The `get_user_similar_movies()` function retrieves the movies rated by both a target user and one of their similar users. It performs an inner merge on `movieId` and appends movie metadata (title and release year) for interpretability. This step supports recommendation insights by identifying overlap in user preferences.

# %% [markdown]
# ### Comparing Ratings Between Users
# 
# The function `get_user_similar_movies(user1, user2, ratings_df, movies_df)` extracts all movies rated by both users by performing an inner join on `movieId`, and merges this with movie titles and release years.

# %%
movies1

# %%
def get_user_similar_movies(user1, user2, ratings_df, movies_df):
    df = ratings_df.copy()
    
    # Ratings for both users
    user1_ratings = df[df['userId'] == user1]
    user2_ratings = df[df['userId'] == user2]
    
    # Movies both users have rated
    common_movies = user1_ratings.merge(user2_ratings, on='movieId', suffixes=('_user1', '_user2'))
    
    # Merge with movie titles and release year
    common_movies = common_movies.merge(movies_df[['movieId', 'title', 'release_year']], on='movieId', how='left')
    
    return common_movies

# %% [markdown]
# ### Comparing Ratings Between Users
# 
# The function `get_user_similar_movies(user1, user2, ratings_df, movies_df)` extracts all movies rated by both users by performing an inner join on `movieId`, and merges this with movie titles and release years.

# %%
get_user_similar_movies(18127, 9197, ratings1, movies1)

# %% [markdown]
# 
# 
# Then there have been selected the top 15 movies rated highest by `user1` (`rating_user1`) and reshaped the data using `melt()` for visualization. The following barplot displays both users' ratings side-by-side for each common movie, allowing visual inspection of agreement or disagreement in preferences.
# 

# %%
common_movies_df = get_user_similar_movies(18127, 9197, ratings1, movies1)

# %%
# Prepare the input dataframe
df_common = common_movies_df[['title', 'rating_user1', 'rating_user2']]

# Columns: ['title', 'rating_user1', 'rating_user2']
df_common = df_common.sort_values('rating_user1', ascending=False).head(15)

# Melt for seaborn compatibility
df_melted_users = df_common.melt(id_vars='title', value_vars=['rating_user1', 'rating_user2'],
                           var_name='User', value_name='Rating')
df_melted_users['User'] = df_melted_users['User'].replace({'rating_user1': 'User 18127', 'rating_user2': 'User 9197'})

# Set the style and palette
sns.set(style="whitegrid")
palette = sns.color_palette("Reds", n_colors=2)

# Create the plot
plt.figure(figsize=(12, 7))
sns.barplot(
    data=df_melted_users,
    y='title',
    x='Rating',
    hue='User',
    palette=palette
)

plt.title('Top 15 Movies Rated by Both Users', fontsize=16, weight='bold')
plt.xlabel('Rating')
plt.ylabel('Movie Title')
plt.legend(title='User', loc='lower right')
plt.legend(title='User', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

plt.tight_layout()
plt.show()


# %% [markdown]
# The strong alignment across most titles like *The Godfather*, *Casablanca*, and *Hoosiers*, indicates a high similarity in taste, consistent with earlier cosine similarity results.
# 
# There are a few divergences, notably in *Aliens*, *Young Frankenstein*, and *Annie Hall*, where ratings differ by 1 or more points. These outliers hint at minor genre preference variations but do not outweigh the overall agreement.
# 
# From a recommender system perspective, this overlap supports the approach of using adjusted ratings and cosine similarity to identify similar users. High agreement across many titles suggests that movies liked by user 9197, but unseen by 18127, are strong candidates for recommendation.
# 

# %% [markdown]
# ### Top Movie Recomendations for User 1827
# 
# To generate personalized movie suggestions, a function named `recommend_movies_from_neighbor()` was applied. This function recommends unseen films to a given user (in this case, 18127) based on the preferences of a similar user 9197, who was identified earlier as one of the top similar neighbors based on cosine similarity.
# 
# The logic begins by collecting all the movies that user 9197 has rated with a score equal to or above 4.0, under the assumption that these represent strong preferences. It then filters out any of these movies that have already been rated by user 18127, ensuring the final list includes only new potential discoveries. These remaining titles are then merged with the `movies1` dataset to retrieve human-readable titles and release years. Finally, the output is sorted by the neighbor’s rating in descending order, and the top 10 recommendations are returned.
# 
# This approach exemplifies classic neighborhood-based collaborative filtering, where the power of peer similarity is used to infer new content preferences.

# %%
def recommend_movies_from_neighbor(user1, user2, ratings_df, movies_df, top_n=10, min_rating=4.0):
    # Ratings from both users
    u1_movies = ratings_df[ratings_df['userId'] == user1]['movieId']
    u2_ratings = ratings_df[(ratings_df['userId'] == user2) & (ratings_df['rating'] >= min_rating)]

    # Keep only movies user1 hasn't seen
    unseen = u2_ratings[~u2_ratings['movieId'].isin(u1_movies)]

    # Merge with movie titles
    recommendations = unseen.merge(movies_df[['movieId', 'title', 'release_year']], on='movieId')

    # Sort by rating and return top N
    return recommendations.sort_values(by='rating', ascending=False).head(top_n)


# %%
recommendations = recommend_movies_from_neighbor(18127, 9197, ratings1, movies1, top_n=10)
recommendations

# %%
# Plot
sns.set(style="whitegrid")
palette = sns.color_palette("Reds", n_colors=1)

plt.figure(figsize=(12, 7))
sns.barplot(data=recommendations, y='title', x='rating', palette=palette)
plt.title('Top 10 Movie Recommendations from Similar User (User 9197)', fontsize=16, weight='bold')
plt.xlabel('Rating')
plt.ylabel('Movie Title')
plt.xlim(0, 5.5)
plt.tight_layout()
plt.show()


# %% [markdown]
# The system recommends 10 unseen films highly rated by a similar user. This includes classics like “It’s a Wonderful Life” and “To Catch a Thief,” as well as critically acclaimed miniseries like “Band of Brothers.” These choices reflect User 9197’s consistent high ratings for vintage dramas, family-friendly narratives, and historical fiction, providing a strong rationale for assuming that User 18127 may appreciate these titles too. Such recommendations can enhance engagement by exploring both personal taste overlap and peer-influenced preferences. 

# %% [markdown]
# # Item-Item Collaborative FIltering

# %% [markdown]
# This approach starts by normalizing user ratings (actual rating minus user mean), removing individual rating bias. A movie-user matrix is then created using these adjusted ratings, with movies as rows and users as columns.
# 
# Unlike the previous user-user model (which compared users), this method compares movies based on how users rated them. This allows us to find similar items and generate recommendations from movies the user already liked.

# %%
# Compute each user's mean rating
user_mean_rating = ratings1.groupby('userId')['rating'].mean().to_frame(name='mean_rating')

# Merge back to ratings to calculate adjusted rating
ratings_merged = pd.merge(ratings1, user_mean_rating, on='userId')
ratings_merged['adjusted_rating'] = ratings_merged['rating'] - ratings_merged['mean_rating']

# Pivot to create the item-user adjusted rating matrix
item_user_matrix = ratings_merged.pivot_table(
    index='movieId', columns='userId', values='adjusted_rating', fill_value=0
)
item_user_matrix.head()


# %% [markdown]
# ### Item-Item Similarity Matrix
# 
# In this step, the pivot matrix `movie_user_adj_matrix` is created to align with the item-item collaborative filtering approach. Unlike the user-user method, this pivot uses `movieId` as the index and `userId` as columns, with values being the adjusted user ratings. This structure enables the comparison of movies based on overlapping user preferences.
# 
# The cosine similarity is then applied across movie vectors (rows) to compute how similar each movie is to every other, based on how users rated them. The diagonal is set to zero to avoid self-similarity. The resulting matrix `item_similarity_df` serves as the core similarity lookup table for building item-based recommendations.
# 

# %%
movie_user_adj_matrix = user_ratings_merged.pivot_table(
    index='movieId',
    columns='userId',
    values='Adjusted-Rating',
    fill_value=0
)

# %%

# Compute item-item (movie-movie) similarity
item_similarity = cosine_similarity(movie_user_adj_matrix)
np.fill_diagonal(item_similarity, 0)

# Convert to DataFrame for readability
item_similarity_df = pd.DataFrame(item_similarity, 
                                   index=movie_user_adj_matrix.index,
                                   columns=movie_user_adj_matrix.index)

item_similarity_df.head()


# %% [markdown]
# This function ranks the most similar movies for every title based on cosine similarity values. For each movie, the top 10 highest-scoring entries (excluding itself) are selected. The resulting `top_10_similar_movies` DataFrame maps each movie to its most similar alternatives—forming the basis for item-based recommendations.
# 

# %%
# Function to get top-N most similar movies for each movie
def find_similar_movies(sim_df, n=10):
    return sim_df.apply(
        lambda row: pd.Series(
            row.sort_values(ascending=False).iloc[:n].index,
            index=[f'top{i+1}' for i in range(n)]
        ),
        axis=1
    )

# Get top 10 similar movies for each movie
top_10_similar_movies = find_similar_movies(item_similarity_df, n=10)
top_10_similar_movies.head()


# %% [markdown]
# This function builds personalized recommendations by identifying movies similar to those a user has rated highly. It looks up the top similar titles for each positively rated movie, aggregates the similarities, and excludes already-rated titles. The result is a sorted list of recommended movies for the user based on item-item similarity.

# %%
def item_item_recommendations(user_id, ratings_df, sim_df, top_n=10, min_rating=4.0):
    # Get movies the user rated highly
    user_rated = ratings_df[(ratings_df['userId'] == user_id) & (ratings_df['rating'] >= min_rating)]

    # Initialize empty list for collecting similar movies
    similar_movies_list = []

    for movie_id in user_rated['movieId']:
        if movie_id in sim_df.index:
            similar_movies_list.append(sim_df.loc[movie_id])

    # Concatenate all similar movie series into one
    if similar_movies_list:
        similar_movies = pd.concat(similar_movies_list)
        similar_movies = similar_movies.groupby(similar_movies.index).mean()
        similar_movies = similar_movies[~similar_movies.index.isin(user_rated['movieId'])]
        return similar_movies.sort_values(ascending=False).head(top_n)
    else:
        return pd.Series(dtype='float64')  # return empty if no matches



# %%
# Generate top 10 item-item recommendations for user 18127
item_recs = item_item_recommendations(18127, ratings1, item_similarity_df, top_n=10)
item_recs


# %%
# Show only the recommended movie titles
recommended_titles = movies1[movies1['movieId'].isin(item_recs.index)]
recommended_titles = recommended_titles.set_index('movieId').loc[item_recs.index]
recommended_titles['Similarity Score'] = item_recs.values

# Output just the titles
recommended_titles['title']



# %% [markdown]
# ### Top Recommended Movies
# 
# The chart displays the top 10 movie recommendations for the User 18127. The similarity scores have been computed with `cosine_similarity` on the `movie_user_adj_matrix`, and the most similar unrated movies are identified.
# 
# To visually emphasize similarity, a normalized score (`mcolors.Normalize`) is mapped to a `Reds` colormap, with higher similarity (darker red) indicating stronger recommendation strength. This custom gradient replaces the default `sns.color_palette()` to better reflect relative importance.

# %%
# Normalize similarity scores for color mapping
norm = mcolors.Normalize(vmin=recommended_titles['Similarity Score'].min(),
                         vmax=recommended_titles['Similarity Score'].max())
colors = [cm.Reds(norm(score)) for score in recommended_titles['Similarity Score']]

# Set style
sns.set(style="whitegrid")

# Create horizontal barplot
plt.figure(figsize=(12, 6))
sns.barplot(
    data=recommended_titles.reset_index(),
    y='title',
    x='Similarity Score',
    palette=colors
)

# Title and axis formatting
plt.title('Top Recommended Movies (Item-Item)', fontsize=16, weight='bold')
plt.xlabel('Similarity Score')
plt.ylabel('Movie Title')
plt.tight_layout()
plt.show()

# %% [markdown]
# Notably, *One Flew Over the Cuckoo’s Nest* and *Monty Python and the Holy Grail* appear at the top with the highest scores, suggesting they share strong rating behavior patterns with movies the user liked. This approach prioritizes movie similarity over user profiles, offering interpretable, content-based suggestions.
# 

# %% [markdown]
# ---

# %%
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids


# %%
# --- K-Means Plot Function ---
def plot_kmeans(x_col, y_col, k):
    df_clean = df_movies_clean[[x_col, y_col]].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(X_scaled)
    df_clean['Cluster'] = labels
    centroids = scaler.inverse_transform(model.cluster_centers_)

    fig = px.scatter(
        df_clean, x=x_col, y=y_col,
        color=df_clean['Cluster'].astype(str),
        title=f"K-Means Clustering (k={k})",
        height=550
    )
    fig.add_scatter(
        x=centroids[:, 0], y=centroids[:, 1],
        mode='markers',
        marker=dict(color='black', size=14, symbol='x'),
        name='Centroids'
    )
    return pn.pane.Plotly(fig, config={'responsive': True}, sizing_mode='stretch_width')


# --- K-Medoids Plot Function ---
def plot_kmedoids(x_col, y_col, k):
    df_clean = df_movies_clean[[x_col, y_col]].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    model = KMedoids(n_clusters=k, random_state=42)
    labels = model.fit_predict(X_scaled)
    df_clean['Cluster'] = labels
    centroids = scaler.inverse_transform(model.cluster_centers_)

    fig = px.scatter(
        df_clean, x=x_col, y=y_col,
        color=df_clean['Cluster'].astype(str),
        title=f"K-Medoids Clustering (k={k})",
        height=550
    )
    fig.update_traces(marker=dict(size=6, opacity=0.6))
    fig.add_scatter(
        x=centroids[:, 0], y=centroids[:, 1],
        mode='markers',
        marker=dict(color='black', size=14, symbol='x'),
        name='Medoids'
    )
    return pn.pane.Plotly(fig, config={'responsive': True}, sizing_mode='stretch_width')


# --- Ratings Overview Dashboard ---
ratings_overview_dashboard = pn.Column(
    pn.pane.Markdown("# Ratings Data Overview"),
    pn.pane.Plotly(fig_day, config={'responsive': True}, sizing_mode='stretch_width'),
    pn.pane.Plotly(fig_month, config={'responsive': True}, sizing_mode='stretch_width')
)

# --- Yearly Ratings Dashboard ---
year_metric_selector = pn.widgets.Select(
    name='Select Metric',
    options=['Rating Count', 'Average Rating'],
    value='Rating Count'
)

yearly_ratings_dashboard = pn.Column(
    pn.pane.Markdown("# Yearly Ratings Analysis"),
    pn.Column(year_metric_selector, sizing_mode='stretch_width', max_width=300),
    pn.bind(plot_by_metric, year_metric_selector)
)

# --- Genre Trends Dashboard ---
genre_trends_dashboard = pn.Column(
    pn.pane.Markdown("# Genre Rating Trends Over Time"),
    pn.Column(genre_selector, year_slider, sizing_mode='stretch_width', max_width=500),
    pn.bind(plot_genre_trends, genre_selector, year_slider)
)

# --- K-Means Dashboard ---
x_axis_kmeans = pn.widgets.Select(name='X-Axis', options=numeric_features, value='avg_rating_movie')
y_axis_kmeans = pn.widgets.Select(name='Y-Axis', options=numeric_features, value='rating_trend')
k_slider_kmeans = pn.widgets.IntSlider(name='Number of Clusters (k)', start=2, end=8, value=3)

kmeans_dashboard = pn.Column(
    pn.pane.Markdown("# K-Means Clustering"),
    pn.Row(
        pn.Column(x_axis_kmeans, y_axis_kmeans, k_slider_kmeans, sizing_mode='stretch_width', max_width=300),
        pn.bind(plot_kmeans, x_axis_kmeans, y_axis_kmeans, k_slider_kmeans)
    )
)

# --- K-Medoids Dashboard ---
x_axis_kmedoids = pn.widgets.Select(name='X-Axis', options=numeric_features, value='avg_rating_movie')
y_axis_kmedoids = pn.widgets.Select(name='Y-Axis', options=numeric_features, value='rating_trend')
k_slider_kmedoids = pn.widgets.IntSlider(name='Number of Clusters (k)', start=2, end=8, value=3)

kmedoids_dashboard = pn.Column(
    pn.pane.Markdown("# K-Medoids Clustering"),
    pn.Row(
        pn.Column(x_axis_kmedoids, y_axis_kmedoids, k_slider_kmedoids, sizing_mode='stretch_width', max_width=300),
        pn.bind(plot_kmedoids, x_axis_kmedoids, y_axis_kmedoids, k_slider_kmedoids)
    )
)

# --- Fuzzy C-Means Dashboard ---
fuzzy_dashboard = pn.Column(
    pn.pane.Markdown("# Fuzzy C-Means Clustering"),
    pn.Column(x_selector, y_selector, sizing_mode='stretch_width', max_width=300),
    pn.bind(plot_fuzzy_clusters, x_selector, y_selector)
)


# %%
pn.extension('plotly')

final_tabs = pn.Tabs(
    ("Genre-Year Heatmap", genre_heatmap_dashboard),
    ("Ratings Overview", ratings_overview_dashboard),
    ("Yearly Ratings", yearly_ratings_dashboard),
    ("Genre Trends", genre_trends_dashboard),
    ("K-Means Clustering", kmeans_dashboard),
    ("K-Medoids Clustering", kmedoids_dashboard),
    ("Fuzzy C-Means Clustering", fuzzy_dashboard),
    dynamic=True,
    sizing_mode='stretch_width',
    tabs_location='above'
)

final_tabs.show()


# %%
# Save to a standalone HTML file
final_tabs.save("CA2_SBA23320_Interactive_Dashboard.html", embed=True)


# %% [markdown]
# - https://www.unixtimestamp.com/
# - https://dash.plotly.com/
# - https://panel.holoviz.org/
# - https://www.w3schools.com/python/pandas/ref_df_explode.asp
# - https://dl.acm.org/doi/pdf/10.1145/2843948 
# - https://www.researchgate.net/publication/227268858_Recommender_Systems_Handbook 
# - https://pmc.ncbi.nlm.nih.gov/articles/PMC7288198/
# - Week 6 – *Clustering 1* (Lecture Notes, 2025), CCT College Dublin.
# - Aurélien Géron (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. O’Reilly Media.
# - https://towardsdatascience.com/silhouette-coefficient-validating-clustering-techniques-e976bb81d10c
# - https://www.sciencedirect.com/science/article/abs/pii/0098300484900207 
# - https://journaljsrr.com/index.php/JSRR/article/view/1812/3576 
# 

# %% [markdown]
# 

# %% [markdown]
# 

# %% [markdown]
# 

# %% [markdown]
# 

# %% [markdown]
# 

# %% [markdown]
# 

# %% [markdown]
# 

# %% [markdown]
# 

# %% [markdown]
# 

# %% [markdown]
# 

# %% [markdown]
# 

# %% [markdown]
# 

# %% [markdown]
# 


