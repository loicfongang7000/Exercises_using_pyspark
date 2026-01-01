from pyspark.sql import SparkSession, DataFrame 
from pyspark.sql.functions import min, max, col, coalesce, lit


import random

# question 1.1 a)

"""
    Génère une liste de triplets (id, value1, value2)

    - id : entier unique de 0 à n
    - value1 : entier entre 0 et 50
    - value2 : entier entre 0 et 5
    """
def generer_liste_triplets(n):
    

    ids = list(range(n+1))          # ids uniques de 0 à n
    random.shuffle(ids)              # mélange pour le côté aléatoire

    triplets = []

    for i in ids:
        value1 = random.randint(0, 50)
        value2 = random.randint(0, 5)
        triplets.append((i, value1, value2))

    return triplets

liste_triplets = generer_liste_triplets(10)

print(liste_triplets)

# question 1.1 b)

#création de Session Spark
spark = SparkSession.builder \
    .appName("PySparkHomework") \
    .master("local[*]") \
    .getOrCreate()

def init_df(nb_rows:int) -> DataFrame:

    data = generer_liste_triplets(nb_rows-1)
    df = spark.createDataFrame(data,["id", "value1", "value2"])     #Génération de Dataframe aléatoire 

    return df

df_aleatoire = init_df(10)  

df_aleatoire.show()

# question 1.1 c)

def transformation(df: DataFrame) -> DataFrame:
    # Normalisation de value1
    stats = df.agg(
        min("value1").alias("min_v"),
        max("value1").alias("max_v")
    ).collect()[0]

    min_v = stats["min_v"]
    max_v = stats["max_v"]

    df = df.withColumn(
        "value1",
        (col("value1") - min_v) / (max_v - min_v)
    )

    # Suppression des value2 = 0
    df = df.filter(col("value2") != 0)

    # Ajout du ratio
    df = df.withColumn(
        "ratio",
        col("value1") / col("value2")
    )

    # Médiane du ratio
    median_ratio = df.approxQuantile("ratio", [0.5], 0.0)[0]

    # Filtrage 
    df = df.filter(col("ratio") >= median_ratio)

    return df

df_transfomation = transformation(df_aleatoire)  

df_transfomation.show()

# question 1.1 d)

def merge(df1: DataFrame, df2: DataFrame) -> DataFrame:

    # Jointure complète sur id
    joined = df1.alias("df1").join(
        df2.alias("df2"),
        on="id",
        how="outer"
    )

    # Addition des colonnes (null -> 0)
    result = joined.select(
        col("id"),

        (
            coalesce(col("df1.value1"), lit(0)) +
            coalesce(col("df2.value1"), lit(0))
        ).alias("value1"),

        (
            coalesce(col("df1.value2"), lit(0)) +
            coalesce(col("df2.value2"), lit(0))
        ).alias("value2")
    )

    # Recalcul du ratio
    result = result.withColumn(
        "ratio",
        col("value1") / col("value2")
    )

    return result



x = transformation(init_df(10))

y= merge(init_df(10), x) #fusion du Dataframe init_df(10) et du dataframe x
y.show()