
import pandas as pd
import panel as pn
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from fcmeans import FCM

pn.extension('plotly')

# Load preprocessed datasets
df_movies_clean1 = pd.read_csv("df_movies_clean1.csv")
ratings1 = pd.read_csv("ratings1.csv")
movies1 = pd.read_csv("movies1.csv")

# -------------- WIDGETS AND FEATURES --------------
numeric_features = [
    'avg_rating_movie', 'num_ratings_movie', 'rating_std_movie',
    'rating_year_range', 'rating_trend', 'genre_count', 'release_decade'
]

# -------------- RATINGS OVERVIEW DASHBOARD --------------
fig_day = px.histogram(ratings1, x="timestamp", nbins=30, title="Ratings by Day")
fig_month = px.histogram(ratings1, x=pd.to_datetime(ratings1['timestamp'], unit='s').dt.to_period("M").astype(str), 
                         title="Ratings by Month")

ratings_overview_dashboard = pn.Column(
    pn.pane.Markdown("## Ratings Data Overview"),
    pn.pane.Plotly(fig_day, config={'responsive': True}, sizing_mode='stretch_width'),
    pn.pane.Plotly(fig_month, config={'responsive': True}, sizing_mode='stretch_width')
)

# -------------- K-MEANS DASHBOARD --------------
def plot_kmeans(x_col, y_col, k):
    df_clean = df_movies_clean1[[x_col, y_col]].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(X_scaled)
    df_clean['Cluster'] = labels
    centroids = scaler.inverse_transform(model.cluster_centers_)

    fig = px.scatter(
        df_clean, x=x_col, y=y_col, color=df_clean['Cluster'].astype(str),
        title=f"K-Means Clustering (k={k})"
    )
    fig.add_scatter(
        x=centroids[:, 0], y=centroids[:, 1],
        mode='markers',
        marker=dict(color='black', size=14, symbol='x'),
        name='Centroids'
    )
    return pn.pane.Plotly(fig, config={'responsive': True}, sizing_mode='stretch_width')

x_axis_kmeans = pn.widgets.Select(name='X-Axis', options=numeric_features, value='avg_rating_movie')
y_axis_kmeans = pn.widgets.Select(name='Y-Axis', options=numeric_features, value='rating_trend')
k_slider_kmeans = pn.widgets.IntSlider(name='Number of Clusters (k)', start=2, end=8, value=3)

kmeans_dashboard = pn.Column(
    pn.pane.Markdown("## K-Means Clustering"),
    pn.Row(
        pn.Column(x_axis_kmeans, y_axis_kmeans, k_slider_kmeans, sizing_mode='stretch_width', max_width=300),
        pn.bind(plot_kmeans, x_axis_kmeans, y_axis_kmeans, k_slider_kmeans)
    )
)

# -------------- K-MEDOIDS DASHBOARD --------------
def plot_kmedoids(x_col, y_col, k):
    df_clean = df_movies_clean1[[x_col, y_col]].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    model = KMedoids(n_clusters=k, random_state=42)
    labels = model.fit_predict(X_scaled)
    df_clean['Cluster'] = labels
    centroids = scaler.inverse_transform(model.cluster_centers_)

    fig = px.scatter(
        df_clean, x=x_col, y=y_col, color=df_clean['Cluster'].astype(str),
        title=f"K-Medoids Clustering (k={k})"
    )
    fig.update_traces(marker=dict(size=6, opacity=0.6))
    fig.add_scatter(
        x=centroids[:, 0], y=centroids[:, 1],
        mode='markers',
        marker=dict(color='black', size=14, symbol='x'),
        name='Medoids'
    )
    return pn.pane.Plotly(fig, config={'responsive': True}, sizing_mode='stretch_width')

x_axis_kmedoids = pn.widgets.Select(name='X-Axis', options=numeric_features, value='avg_rating_movie')
y_axis_kmedoids = pn.widgets.Select(name='Y-Axis', options=numeric_features, value='rating_trend')
k_slider_kmedoids = pn.widgets.IntSlider(name='Number of Clusters (k)', start=2, end=8, value=3)

kmedoids_dashboard = pn.Column(
    pn.pane.Markdown("## K-Medoids Clustering"),
    pn.Row(
        pn.Column(x_axis_kmedoids, y_axis_kmedoids, k_slider_kmedoids, sizing_mode='stretch_width', max_width=300),
        pn.bind(plot_kmedoids, x_axis_kmedoids, y_axis_kmedoids, k_slider_kmedoids)
    )
)

# -------------- FUZZY C-MEANS DASHBOARD --------------
axis_options = numeric_features
x_selector = pn.widgets.Select(name='X-Axis', options=axis_options, value='avg_rating_movie')
y_selector = pn.widgets.Select(name='Y-Axis', options=axis_options, value='num_ratings_movie')

def plot_fuzzy_clusters(x, y):
    fig = px.scatter(
        df_movies_clean1,
        x=x, y=y,
        color=df_movies_clean1['fuzzy_cluster'].astype(str),
        size='fuzzy_membership',
        opacity=0.65,
        title=f'Fuzzy C-Means Clustering: {x} vs {y}',
        labels={'color': 'Cluster'}
    )
    return pn.pane.Plotly(fig, config={'responsive': True}, sizing_mode='stretch_width')

fuzzy_dashboard = pn.Column(
    pn.pane.Markdown("## Fuzzy C-Means Clustering"),
    pn.Row(x_selector, y_selector, sizing_mode='stretch_width'),
    pn.bind(plot_fuzzy_clusters, x_selector, y_selector)
)

# -------------- FINAL TABS --------------
final_tabs = pn.Tabs(
    ("Ratings Overview", ratings_overview_dashboard),
    ("K-Means Clustering", kmeans_dashboard),
    ("K-Medoids Clustering", kmedoids_dashboard),
    ("Fuzzy C-Means Clustering", fuzzy_dashboard),
    dynamic=True,
    sizing_mode='stretch_width',
    tabs_location='above'
)

final_tabs.servable()
