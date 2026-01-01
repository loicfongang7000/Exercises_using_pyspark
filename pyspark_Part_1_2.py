#Import des Biliotheques utiles
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, split, regexp_extract, when, to_date, explode, count, dense_rank, sum, countDistinct, year, avg, size
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler , StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.sql.types import DoubleType
from pyspark.ml.clustering import KMeans

# Question 1.1 a)

#Creation de d'une Session Spark
spark = SparkSession.builder \
    .appName("PySpark_netflix") \
    .master("local[*]") \
    .getOrCreate()

#Considerer la premiere lignes comme le nom des colonnes. Utiliser les donnes pour deviner le datatype.  
#Lire les fichier csv est tranformer les donnees en dataframe.

df = spark.read.option("header", "true") \
    .option("inferSchema", "true") \
    .csv("netflix.csv")

#creation de la colonne duration minutes avec les "durations" qui sont en minutes
df = df.withColumn(
    "duration_min",
    when(
        col("duration").contains("min"),
        regexp_extract(col("duration"), r"(\d+)", 1).cast("int")
    )
)

#creation de la colonne duration season  avec les "durations" qui sont en saisons
df = df.withColumn(
    "duration_season",
    when(
        col("duration").contains("Season"),
        regexp_extract(col("duration"), r"(\d+)", 1).cast("int")
    )
)

#Changer le type de la colonne date_added au type : date 
df = df.withColumn(
    "date_added",
    to_date(col("date_added"), "MMMM d, yyyy")
)

#normalisation des catégories dans listed_in en une liste (array de strings)
df = df.withColumn(
    "listed_in",
    split(col("listed_in"), ", ")
)

df.printSchema()  # Types de variables des differentes colonnes du DataFrame

df.select(
    "type", "duration", "duration_min", "duration_season",
    "date_added", "listed_in"
).show(truncate=False)

#Question 1.1 b)

#Function qui retourne les 3 pays avec le plus de titres
def top_countries(df: DataFrame) -> DataFrame:
    
    # Séparer les pays et exploser
    df_country = (
        df
        .withColumn("country", split(col("country"), ", "))
        .withColumn("country", explode(col("country")))
    )

    #Compter le nombre de titres par type et pays
    df_count = (
        df_country
        .groupBy("type", "country")
        .agg(count("*").alias("nb_titles"))
    )

    #Définir la fenêtre de ranking
    window_spec = Window.partitionBy("type").orderBy(col("nb_titles").desc())

    #Appliquer dense_rank
    df_ranked = df_count.withColumn(
        "rank",
        dense_rank().over(window_spec)
    )

    #Garder les 3 premiers pays par type
    result = df_ranked.filter(col("rank") <= 3)

    return result

df_clean = df.filter(
    col("type").isin("Movie", "TV Show")
)

top3 = top_countries(df_clean)
top3.show(truncate=False)

# Question 1.1 c)

def actor_stats(df: DataFrame) -> DataFrame:

    #Explosion des acteurs
    df_actor = (
        df
        .withColumn("actor", split(col("cast"), ", "))
        .withColumn("actor", explode(col("actor")))
    )

    #Comptages des contenus, films et séries
    stats_basic = (
        df_actor
        .groupBy("actor")
        .agg(
            countDistinct("show_id").alias("total_contents"),
            sum(when(col("type") == "Movie", 1).otherwise(0)).alias("nb_movies"),
            sum(when(col("type") == "TV Show", 1).otherwise(0)).alias("nb_tv_shows")
        )
    )

    #Explosion des genres pour la diversité
    df_actor_genre = df_actor.withColumn("genre", explode(col("listed_in")))

    stats_genre = (
        df_actor_genre
        .groupBy("actor")
        .agg(countDistinct("genre").alias("genre_diversity"))
    )

    #Jointure pour tout combiner
    stats = stats_basic.join(stats_genre, on="actor", how="left")

    return stats


actor_df = actor_stats(df_clean)
actor_df.show(truncate=False)

# Question 1.1 d)

#Top 10 des couples d'acteurs qui collaborent frequemment

def actor_pairs(df: DataFrame) -> DataFrame:
    
    #Explosion des acteurs
    df_actor = df.withColumn("actor", split(col("cast"), ", ")) \
                 .withColumn("actor", explode(col("actor")))
    
    #CrossJoin sur show_id pour former tous les couples
    df_pairs = df_actor.alias("a").join(
        df_actor.alias("b"),
        on="show_id"
    )
    
    #Éviter auto-collaboration et doublons
    df_pairs = df_pairs.filter(col("a.actor") < col("b.actor"))
    
    #Compter le nombre de collaborations
    df_pairs_count = df_pairs.groupBy(
        col("a.actor").alias("actor_1"),
        col("b.actor").alias("actor_2")
    ).agg(
        count("*").alias("nb_collaborations")
    )
    
    #Top 10
    top10_pairs = df_pairs_count.orderBy(col("nb_collaborations").desc()).limit(10)
    
    return top10_pairs

top_collabs = actor_pairs(df_clean)
top_collabs.show(truncate=False)

# Question 1.1 e)

def time_analysis(df: DataFrame) -> DataFrame:
    

    # Nettoyage
    df_cleaned = df.filter(col("date_added").isNotNull())

    #Moyenne annuelle des nouvelles productions (release_year)
    df_annual_prod = df_cleaned.groupBy("release_year").agg(count("*").alias("nb_releases"))
    avg_annual = df_annual_prod.agg(avg("nb_releases").alias("avg_productions")).collect()[0][0]

    #Nombre de contenus ajoutés par année
    df_yearly = df_cleaned.withColumn("year", year(col("date_added"))) \
                          .groupBy("year") \
                          .agg(count("*").alias("nb_added"))

    #Vectorisation pour ML
    assembler = VectorAssembler(inputCols=["year"], outputCol="features")
    df_yearly_feat = assembler.transform(df_yearly)

    #Régression linéaire (croissance/décroissance)
    lr = LinearRegression(featuresCol="features", labelCol="nb_added")
    lr_model = lr.fit(df_yearly_feat)
    df_pred = lr_model.transform(df_yearly_feat) \
                      .withColumnRenamed("prediction", "regression_trend")

    #Ajouter la moyenne annuelle comme constante
    df_final = df_pred.withColumn("avg_productions", col("regression_trend")*0 + avg_annual) \
                      .select("year", "nb_added", "regression_trend") \
                      .orderBy("year")

    return df_final


df_time = time_analysis(df_clean)
df_time.show(truncate=False)

def cluster_contents(df: DataFrame) -> DataFrame:

    #Fonction utilitaire pour nettoyer les colonnes numériques
    def clean_double(df, col_name):
        return df.withColumn(
            col_name,
            when(col(col_name).rlike("^[0-9]+$"), col(col_name).cast(DoubleType()))
            .otherwise(0.0)
        )
    
    #Nettoyage des colonnes
    df_cleaned = df

    # duration_min : garder uniquement les nombres, sinon 0
    df_cleaned = clean_double(df_cleaned, "duration_min")

    # release_year : garder uniquement les nombres, sinon 0
    df_cleaned = clean_double(df_cleaned, "release_year")

    # size_cast : nombre d'acteurs, 0 si cast vide ou null
    df_cleaned = df_cleaned.withColumn(
        "size_cast",
        when(col("cast").isNotNull(), size(split(col("cast"), ", "))).otherwise(0.0).cast(DoubleType())
    )

    #VectorAssembler
    assembler = VectorAssembler(
        inputCols=["duration_min", "release_year", "size_cast"],
        outputCol="features_vec"
    )
    df_vect = assembler.transform(df_cleaned)

    #StandardScaler
    scaler = StandardScaler(
        inputCol="features_vec",
        outputCol="features_scaled",
        withMean=True,
        withStd=True
    )
    scaler_model = scaler.fit(df_vect)
    df_scaled = scaler_model.transform(df_vect)

    #K-Means clustering
    kmeans = KMeans(featuresCol="features_scaled", predictionCol="cluster", k=4, seed=42)
    kmeans_model = kmeans.fit(df_scaled)
    df_clustered = kmeans_model.transform(df_scaled)

    #DataFrame final
    df_final = df_clustered.select(
        "show_id",
        "title",
        "type",
        "duration_min",
        "release_year",
        "size_cast",
        "cluster"
    )

    return df_final

df_clusters = cluster_contents(df)
df_clusters.show(truncate=False)


