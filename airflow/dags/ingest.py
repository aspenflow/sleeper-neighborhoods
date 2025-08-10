from airflow.models import DAG
from airflow.hooks.base import BaseHook
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from sqlalchemy import create_engine
from datetime import datetime
import pandas as pd
from geopy.distance import geodesic
import requests
import statsmodels.api as sm
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator
from sqlalchemy import create_engine, text
from airflow.utils.trigger_rule import TriggerRule


def db_connection():
    c = BaseHook.get_connection('housing')
    url = f'postgresql://{c.login}:{c.password}@{c.host}:{c.port}/{c.schema}'
    engine = create_engine(url)
    return engine


def _area_coords_exists() -> bool:
    engine = db_connection()
    try:
        with engine.connect() as conn:
            return conn.execute(text("SELECT to_regclass('public.area_coordinates') IS NOT NULL")).scalar()
    finally:
        engine.dispose()

def _decide_build_area_key():
    return 'skip_build_area_key' if _area_coords_exists() else 'build_area_key'

def get_coordinates_for(borough):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": f"{borough}, New York, NY",
        "format": "json",
        'limit': 1
    }
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, params=params, headers=headers)
    data = response.json()
    if data:
        print(data[0])
        return float(data[0]['lat']), float(data[0]['lon'])

def get_neighborhood_at(k, lat, lon):
    return min(k.items(), key=lambda x: geodesic((lat, lon), x[1]).meters)[0]

def standardize(data):
    return (data - data.median()) / (data.quantile(0.75) - data.quantile(0.25))


def load_rent_stats():
    df = pd.read_csv(
        '/opt/airflow/data/medianAskingRent_Studio.csv',
        usecols=lambda col: col not in ['Borough', 'areaType']
    )

    # Create table for area stats
    rent_stat = pd.DataFrame()
    rent_stat['areaName'] = df['areaName']

    # Place median across all records for each area in new df with index as area name
    rent_stat['overall_median'] = df.median(axis=1, numeric_only=True)
    rent_stat.dropna(inplace=True)
    rent_stat.set_index('areaName', inplace=True)

    # transpose to add date column to sort by date and get most recent n records
    df = df.T.reset_index()
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True)
    df.rename(columns={'areaName': 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    reversed_df = df.iloc[::-1]

    engine = db_connection()
    # gets median for the maximum 15 most recent points
    recent_measurements = reversed_df.apply(lambda col: col.dropna().head(15), axis=0).drop(columns='date').T
    rent_stat['recent_median'] = recent_measurements.median(axis=1)
    rent_stat.reset_index(inplace=True)
    rent_stat.to_sql('rent_stat', engine, if_exists='replace', index=False)

    engine.dispose()

def load_crime_data():
    engine = db_connection()
    pd.read_csv(
        '/opt/airflow/data/NYPD_Complaint_Data_Current__Year_To_Date__20250410.csv',
        usecols=[
            'Latitude',
            'Longitude'
        ]
    ).to_sql('crime', engine, if_exists='replace', index=False)

    engine.dispose()

def build_area_key():
    engine = db_connection()
    df = pd.read_csv(
            '/opt/airflow/data/medianAskingRent_Studio.csv',
            usecols=['areaName', 'areaType']
    )
    df = df[~df['areaType'].isin(['borough', 'city', 'submarket'])]
    coords = df['areaName'].apply(lambda x: get_coordinates_for(x))
    df['lat'] = coords.apply(lambda x: x[0] if x else None)
    df['lon'] = coords.apply(lambda x: x[1] if x else None)
    df[['areaName', 'areaType', 'lat', 'lon']].to_sql('area_coordinates', engine, if_exists='replace', index=False)
    engine.dispose()

def standardize_db():
    engine = db_connection()

    df = pd.read_sql(
        """
        SELECT  *
        FROM    neighborhood_stats
        """,
        con=engine
    )

    std_cols = ['num_crimes', 'overall_median', 'recent_median', 'avg_noise', 'avg_age', 'median_floors']
    df_ret = df[['area']]
    for col in std_cols:
        df_ret[f'{col}_std'] = standardize(df[col])

    df_ret.to_sql('std', engine, if_exists='replace', index=False)

    engine.dispose()

def anomalies():
    engine = db_connection()
    df = pd.read_sql(
        """
        SELECT  *
        FROM    neighborhood_stats
        """,
        con=engine
    )

    X = sm.add_constant(df[['num_crimes', 'avg_noise', 'avg_age', 'median_floors']])
    y = df['recent_median']

    # Fit the regression model
    model = sm.OLS(y, X).fit()

    # Predict rent using the model
    df['predicted_rent'] = model.predict(X)

    # Compute residuals: actual - predicted
    df['residual'] = df['recent_median'] - df['predicted_rent']

    # flag large negative residuals (e.g., below -1.5 standard deviations)
    threshold = df['residual'].mean() - 1.25 * df['residual'].std()
    df['unexpectedly_low_rent'] = df['residual'] < threshold
    df[['area', 'unexpectedly_low_rent', 'predicted_rent']].to_sql('anomalies', engine, if_exists='replace', index=False)
    print(model.summary())
    engine.dispose()

def load_construction():
    engine = db_connection()
    df = pd.read_csv(
        '/opt/airflow/data/HousingDB_post2010.csv',
                        low_memory=False, usecols=['FloorsProp', 'CompltYear', 'Latitude', 'Longitude']
                     ).to_sql('construction', engine, if_exists='replace', index=False)
    engine.dispose()


def load():
    import statsmodels.api as sm
    import pydeck as pdk
    import matplotlib.pyplot as plt

    engine = db_connection()
    df = pd.read_sql(
        """
        SELECT  *
        FROM    neighborhood_stats
        """,
        con=engine
    )

    # Normalize scores between 0–1
    df['score_norm'] = (df['score'] - df['score'].min()) / (df['score'].max() - df['score'].min())
    df['avg_noise'] = df['avg_noise'].round(1)
    df['score_norm'] = df['score_norm'].round(2)

    # Use a matplotlib colormap to get RGB
    cmap = plt.get_cmap("viridis")  # or 'plasma', 'inferno', 'magma'
    df['color'] = df['score_norm'].apply(lambda x: [int(255 * c) for c in cmap(x)[:3]])

    df['anomaly_color'] = df['unexpectedly_low_rent'].apply(
        lambda x: [50, 225, 25] if x else [200, 200, 200]
    )

    layer = pdk.Layer(
        "ColumnLayer",
        data=df,
        get_position='[lon, lat]',
        get_elevation='score_norm',
        elevation_scale=1200,
        radius=200,
        get_fill_color='anomaly_color',
        pickable=True,
        auto_highlight=True
    )

    view_state = pdk.ViewState(
        latitude=df['lat'].mean(),
        longitude=df['lon'].mean(),
        zoom=10,
        pitch=45
    )

    # Uses restricted API key
    r = pdk.Deck(layers=[layer],
                 initial_view_state=view_state,
                 api_keys={
                     'mapbox': 'pk.eyJ1IjoiamgyOTA0dTkwMjR0IiwiYSI6ImNtZHQ0ZWdxdTEya2EyaXBvcW5uZDZ4ajMifQ.234e5_ej-wo9lP14w6EePg'},
                 map_provider='mapbox',
                 tooltip={
                     "html": "<div><b>Neighborhood:</b> <span>{area}</span><br/>"
                             "<b>Expected Rent ($):</b> <span>{predicted_rent}</span><br/>"
                             "<b>Actual Rent ($):</b> <span>{recent_median}</span><br/>"
                             "<b>Crime Count:</b> <span>{num_crimes}</span><br/>"
                             "<b>Noise Level (dB):</b> <span>{avg_noise}</span><br/>"
                             "<b>Building Floors:</b> <span>{median_floors}</span><br/>"
                             "<b>Building Age:</b> <span>{avg_age}</span><br/>"
                             "<b>Score (Norm):</b> <span>{score_norm}</span></div>"
                 }
                 )
    r.to_html('/opt/airflow/output/index.html', notebook_display=False, open_browser=False)


with DAG(
    dag_id='ingest',
    start_date=datetime(2020, 1, 1),
    schedule_interval = None,
    catchup = False
) as dag:

    load_rent_task = PythonOperator(
        task_id='load_rent_stats',
        python_callable=load_rent_stats
    )

    load_crime_task = PythonOperator(
        task_id='load_crime_data',
        python_callable=load_crime_data
    )

    add_area_spatial_index_task = SQLExecuteQueryOperator(
        task_id='add_area_spatial_index',
        conn_id='housing',
        sql="""
            CREATE EXTENSION IF NOT EXISTS postgis;
            ALTER TABLE area_coordinates
                ADD COLUMN IF NOT EXISTS geom GEOGRAPHY(Point, 4326);
            UPDATE area_coordinates
                SET geom = ST_SetSRID(ST_MakePoint(lon, lat), 4326);
            CREATE INDEX ON area_coordinates USING GIST (geom);
        """
    )

    classify_crime_area_task = SQLExecuteQueryOperator(
        task_id='classify_crime_area',
        conn_id='housing',
        sql=
        """
            ALTER TABLE crime
                ADD COLUMN IF NOT EXISTS geom GEOGRAPHY(Point, 4326);
            UPDATE crime
                SET geom = ST_SetSRID(ST_MakePoint("Longitude", "Latitude"), 4326);
            CREATE INDEX on crime USING GIST (geom);
            
            DROP TABLE IF EXISTS crime_locations;
            CREATE TABLE crime_locations AS 
            SELECT  n.area AS area,
                    crime."Latitude" AS lat,
                    crime."Longitude" AS lon
            FROM    crime
            JOIN LATERAL (
                SELECT  "areaName" AS area, geom
                FROM    area_coordinates
                ORDER BY crime.geom <-> geom
                LIMIT 1
            ) n ON true;
        """
    )

    build_area_key_task = PythonOperator(
        task_id='build_area_key',
        python_callable=build_area_key
    )

    compute_crime_aggregates_task = SQLExecuteQueryOperator(
        task_id='compute_crime_aggregates',
        conn_id='housing',
        sql="""
            DROP TABLE IF EXISTS crime_by_neighborhood;
            CREATE TABLE crime_by_neighborhood AS
            SELECT  a.area AS area,
                    a.num_crimes AS num_crimes,
                    area_coordinates.lat AS lat,
                    area_coordinates.lon AS lon
            FROM (
                SELECT  COUNT(area) AS num_crimes,
                        area
                FROM    crime_locations
                GROUP BY area
            ) a JOIN area_coordinates ON a.area = area_coordinates."areaName"
        """
    )

    join_rent_crime_task = SQLExecuteQueryOperator(
        task_id='join_rent_crime',
        conn_id='housing',
        sql=
            """
            DROP TABLE IF EXISTS neighborhood_stats;
            CREATE TABLE neighborhood_stats AS
            SELECT  area,
                    crime_by_neighborhood.lat,
                    crime_by_neighborhood.lon,
                    num_crimes,
                    rent_stat.overall_median,
                    rent_stat.recent_median
            FROM    crime_by_neighborhood JOIN rent_stat ON crime_by_neighborhood.area = rent_stat."areaName"
            """
    )

    load_construction_task = PythonOperator(
        task_id='load_construction',
        python_callable=load_construction
    )

    classify_construction_task = SQLExecuteQueryOperator(
        task_id='classify_construction_area',
        conn_id='housing',
        sql="""
            ALTER TABLE construction
                ADD COLUMN IF NOT EXISTS geom GEOGRAPHY(Point, 4326);
            UPDATE construction
                SET geom = ST_SetSRID(ST_MakePoint("Longitude", "Latitude"), 4326);
            CREATE INDEX on construction USING GIST (geom);
            
            DROP TABLE IF EXISTS construction_locations;
            CREATE TABLE construction_locations AS 
            SELECT  n.area AS area,
                    construction."Latitude" AS lat,
                    construction."Longitude" AS lon,
                    construction."FloorsProp" AS floors,
                    construction."CompltYear" AS year_complete
            FROM    construction
            JOIN LATERAL (
                SELECT  "areaName" AS area, geom
                FROM    area_coordinates
                ORDER BY construction.geom <-> geom
                LIMIT 1
            ) n ON true;
        """
    )

    construction_aggregates_task = SQLExecuteQueryOperator(
        task_id='construction_aggregates',
        conn_id='housing',
        sql="""
            DROP TABLE IF EXISTS med_stories_by_neighborhood;
            CREATE TABLE med_stories_by_neighborhood AS
            SELECT  a.area AS area,
                    a.median_floors AS median_floors,
                    2025 - a.avg_year AS avg_age,
                    area_coordinates.lat AS lat,
                    area_coordinates.lon AS lon
            FROM (
                SELECT  AVG(year_complete) AS avg_year,
                        percentile_cont(0.5) WITHIN GROUP (ORDER BY floors) AS median_floors,
                        area
                FROM    construction_locations
                GROUP BY area
            ) a JOIN area_coordinates ON a.area = area_coordinates."areaName"
        """
    )

    join_construction_task = SQLExecuteQueryOperator(
        task_id='join_construction',
        conn_id='housing',
        sql="""
        ALTER TABLE neighborhood_stats
            ADD COLUMN IF NOT EXISTS median_floors NUMERIC(10, 2),
            ADD COLUMN IF NOT EXISTS avg_age NUMERIC(10, 2);
            
        UPDATE neighborhood_stats
            SET median_floors = m.median_floors,
                avg_age = m.avg_age
            FROM med_stories_by_neighborhood m
            WHERE neighborhood_stats.area = m.area;
        """
    )

    c = BaseHook.get_connection('housing')
    global PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE
    PGHOST = c.host
    PGPORT = c.port
    PGUSER = c.login
    PGPASSWORD = c.password
    PGDATABASE = c.schema

    assign_shell_role_task = SQLExecuteQueryOperator(
        task_id='assign_shell_role',
        conn_id='housing',
        sql=f"""
            GRANT SELECT, INSERT, UPDATE, DELETE ON ny_noise_raster TO {PGUSER};
        """
    )

    clip_tiff_task = BashOperator(
        task_id='clip_tiff',
        bash_command=(
            "gdalwarp "
            "-t_srs EPSG:5070 "
            "-te -74.3 40.45 -73.65 40.95 "
            "-te_srs EPSG:4326 "
            "-overwrite "
            "-co COMPRESS=DEFLATE "
            "/opt/airflow/data/NY_rail_road_and_aviation_noise_2020.tif "
            "/opt/airflow/data/tmp/NY_noise_clipped.tif"
        )
    )

    load_raster_task = BashOperator(
        task_id='load_raster',
        bash_command=(
            f"raster2pgsql -s 5070 -I -C -M -t 100x100 "
            f"/opt/airflow/data/tmp/NY_noise_clipped.tif "
            f"public.ny_noise_raster | "
            f"PGPASSWORD={PGPASSWORD} psql -h {PGHOST} -p {PGPORT} -U {PGUSER} -d {PGDATABASE}"
        ),
        env={"PGPASSWORD": PGPASSWORD}
    )

    repair_raster_task = SQLExecuteQueryOperator(
        task_id='repair_raster',
        conn_id='housing',
        sql="""
            -- Reproject the NYC bounding box into EPSG:5070
            DROP TABLE IF EXISTS noise;
            CREATE TABLE noise AS
            WITH nyc_bounds AS (
                SELECT ST_Transform(
                    ST_MakeEnvelope(-74.3, 40.45, -73.65, 40.95, 4326), 5070
                ) AS geom
            ),
            clipped_raster AS (
              SELECT ST_Clip(rast, nyc_bounds.geom) AS rast
              FROM ny_noise_raster, nyc_bounds
            ),
            pixels AS (
              SELECT 
                (ST_PixelAsPoints(rast)).val AS val,
                (ST_PixelAsPoints(rast)).geom AS geom
              FROM clipped_raster
            )
            SELECT
                val AS noise_value,
                ST_X(ST_Transform(geom, 4326)) AS lon,
                ST_Y(ST_Transform(geom, 4326)) AS lat
            FROM pixels;
            """
    )

    add_noise_spatial_index_task = SQLExecuteQueryOperator(
        task_id='add_noise_spatial_index',
        conn_id='housing',
        sql="""
            ALTER TABLE noise
                ADD COLUMN IF NOT EXISTS geom GEOGRAPHY(Point, 4326);
            UPDATE noise
                SET geom = ST_SetSRID(ST_MakePoint("lon", "lat"), 4326);
            CREATE INDEX on noise USING GIST (geom);
        """
    )

    add_area_spatial_index_task.trigger_rule = TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS

    classify_noise_area_task = SQLExecuteQueryOperator(
        task_id='classify_noise_area',
        conn_id='housing',
        sql="""
            DROP TABLE IF EXISTS noise_locations;
            CREATE TABLE noise_locations AS 
            SELECT  n.area AS area,
                    noise.lat AS lat,
                    noise.lon AS lon,
                    noise.noise_value
            FROM    noise
            JOIN LATERAL (
                SELECT  "areaName" AS area, geom
                FROM    area_coordinates
                ORDER BY noise.geom <-> geom
                LIMIT 1
            ) n ON true;
        """
    )

    compute_noise_aggregates_task = SQLExecuteQueryOperator(
        task_id='compute_noise_aggregates',
        conn_id='housing',
        sql="""
        DROP TABLE IF EXISTS noise_aggregates;
        CREATE TABLE noise_aggregates AS 
        SELECT  area,
                AVG(noise_value) AS avg_noise
        FROM    noise_locations
        GROUP BY area;
        """
    )

    join_noise_stats_task = SQLExecuteQueryOperator(
        task_id='join_noise_stats',
        conn_id='housing',
        sql="""
        ALTER TABLE neighborhood_stats
        ADD COLUMN avg_noise FLOAT;
        
        UPDATE neighborhood_stats
        SET avg_noise = noise_aggregates.avg_noise
        FROM noise_aggregates
        WHERE noise_aggregates.area = neighborhood_stats.area;
        """
    )

    standardize_task = PythonOperator(
        task_id='standardize',
        python_callable=standardize_db
    )

    append_std_cols_task = SQLExecuteQueryOperator(
        task_id='append_std_cols',
        conn_id='housing',
        sql= """
        ALTER TABLE neighborhood_stats
        ADD COLUMN IF NOT EXISTS num_crimes_std NUMERIC(10,3),
        ADD COLUMN IF NOT EXISTS overall_median_std NUMERIC(10,3),
        ADD COLUMN IF NOT EXISTS recent_median_std NUMERIC(10,3),
        ADD COLUMN IF NOT EXISTS avg_noise_std NUMERIC(10,3),
        ADD COLUMN IF NOT EXISTS avg_age_std NUMERIC(10,3),
        ADD COLUMN IF NOT EXISTS med_floors_std NUMERIC(10,3);
        
        UPDATE neighborhood_stats
        SET
            num_crimes_std = std.num_crimes_std,
            overall_median_std = std.overall_median_std,
            recent_median_std = std.recent_median_std,
            avg_noise_std = std.avg_noise_std,
            avg_age_std = std.avg_age_std,
            med_floors_std = std.median_floors_std
        FROM std
        WHERE neighborhood_stats.area = std.area;
        """
    )

    anomalies_task = PythonOperator(
        task_id='anomalies',
        python_callable=anomalies
    )

    append_anomalies_task = SQLExecuteQueryOperator(
        task_id='append_anomalies',
        conn_id='housing',
        sql="""
        ALTER TABLE neighborhood_stats
        ADD COLUMN IF NOT EXISTS predicted_rent NUMERIC(10,2),
        ADD COLUMN IF NOT EXISTS unexpectedly_low_rent BOOLEAN;
        
        UPDATE neighborhood_stats
        SET
            predicted_rent = anomalies.predicted_rent,
            unexpectedly_low_rent = anomalies.unexpectedly_low_rent
        FROM anomalies
        WHERE anomalies.area = neighborhood_stats.area;
        """
    )

    score_task = SQLExecuteQueryOperator(
        task_id='score',
        conn_id='housing',
        sql="""
        ALTER TABLE neighborhood_stats
        ADD COLUMN IF NOT EXISTS score NUMERIC(10,3);
        
        WITH vars AS (
            SELECT  3 AS w_crime, 3 AS w_noise, 1 AS w_rent, 1 AS w_age, 1 AS w_floors
        )
        UPDATE neighborhood_stats
        SET
            score = w_crime*num_crimes_std +
                    w_noise*avg_noise_std +
                    w_rent*(overall_median_std - recent_median_std)/overall_median_std +
                    w_age*avg_age_std
        FROM vars;
        """
    )

    garbage_collection_task = SQLExecuteQueryOperator(
        task_id='garbage_collection',
        conn_id='housing',
        sql="""
        DROP TABLE IF EXISTS    crime,
                                crime_by_neighborhood,
                                crime_locations,
                                rent,
                                rent_stat,
                                noise,
                                noise_aggregates,
                                noise_locations,
                                ny_noise_raster,
                                std,
                                med_stories_by_neighborhood,
                                construction,
                                construction_locations
                                ;
        """
    )

    build_map_task = PythonOperator(
        task_id='build_map',
        python_callable=load
    )

    branch_area_table = BranchPythonOperator(
        task_id='branch_area_table',
        python_callable=_decide_build_area_key
    )

    skip_build_area_key = EmptyOperator(task_id='skip_build_area_key')

    load_crime_task >> classify_crime_area_task >> compute_crime_aggregates_task
    [load_rent_task, compute_crime_aggregates_task] >> join_rent_crime_task
    load_raster_task >> assign_shell_role_task >> repair_raster_task
    classify_noise_area_task >> compute_noise_aggregates_task
    [compute_noise_aggregates_task, join_rent_crime_task] >> join_noise_stats_task
    repair_raster_task >> add_noise_spatial_index_task >> classify_noise_area_task


    clip_tiff_task >> load_raster_task

    join_noise_stats_task >> anomalies_task >> append_anomalies_task
    join_noise_stats_task >> standardize_task >> append_std_cols_task >> score_task >> garbage_collection_task >> build_map_task

    load_crime_task >> load_construction_task
    load_construction_task >> classify_construction_task >> construction_aggregates_task >> join_rent_crime_task >> join_construction_task

    join_construction_task >> [anomalies_task, standardize_task]

    branch_area_table >> [build_area_key_task, skip_build_area_key]
    [build_area_key_task, skip_build_area_key] >> add_area_spatial_index_task
    add_area_spatial_index_task >> [classify_noise_area_task, classify_construction_task, classify_crime_area_task]