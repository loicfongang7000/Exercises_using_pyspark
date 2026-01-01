#Import des Librairies Necessaires
from pyspark.sql import SparkSession, DataFrame 
from pyspark.sql.functions import col, from_json, explode, regexp_replace
from pyspark.sql.types import ArrayType, StructType, StructField, IntegerType, StringType

# Question 1.1 a)

# Creation Session Spark
spark = SparkSession.builder \
    .appName("PySpark_movies_correction") \
    .master("local[*]") \
    .getOrCreate()


df= spark.read.option("header", "true").option("inferSchema", "true").csv("movies.csv") #Lecture des donnees dans le fichier movies.csv et tranformation en DF spark en considerant la premiere ligne comme noms des colonnes
df.createOrReplaceTempView("movies")  # Creation d'une table temporaire pour permettre interrogation du DataFrame par SQL

df.printSchema()  #Voir le type des donnees des colonnes movies

year = spark.sql("""
    SELECT
        id,
        original_title,
        regexp_extract(release_date, '^(\\\\d{4})', 1) AS year    /*on prend les 4 premier chiffre dans la colonne year */
    FROM movies
""")

# Question 1.1 b)

# Définir le schéma JSON
genre_schema = ArrayType(
    StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True)
    ])
)

# Créer un nouveau DataFrame avec array struct
df_parsed = df.withColumn(
    "genres_parsed",
    from_json(
        regexp_replace(col("genres"), "'", '"'),  # remplacer ' par "
        genre_schema
    )
)

#Les genres les plus frequent
most_pop_movies = (
    df_parsed
        .select(explode(col("genres_parsed")).alias("g"))
        .groupBy(col("g.name").alias("genre"))
        .count()
        .orderBy(col("count").desc())
)

most_pop_movies.show(10)

# Question 1.1 c)

#Les films qui appartiennent a au moins 3 genres.
films_multi_genres = spark.sql("""
    SELECT
        id,
        original_title,
        (
            length(genres) - length(regexp_replace(genres, 'name', ''))
        ) / length('name') AS nb_genres
    FROM movies
    WHERE (
        length(genres) - length(regexp_replace(genres, 'name', ''))
    ) / length('name') >= 3
""")

films_multi_genres.show()

# Question 1.1 d)

year.createOrReplaceTempView("movies_with_year")

#Notre Common Table Expression avec les 5 annees ayant produit le plus de films
CTE = spark.sql("""
    WITH films_par_annee AS (
        SELECT
            year,
            COUNT(*) AS nb_films
        FROM movies_with_year
        WHERE year != '' /* on n'inclut pas le nombre de filme sans annee  */
        GROUP BY year
    )
    SELECT *
    FROM films_par_annee
    ORDER BY nb_films DESC
    LIMIT 5
""")


CTE.show()


# Question 1.1 e)



movies_with_genres = spark.sql("""
    SELECT
        id,
        genres
    FROM movies
""")


classement = spark.sql("""
    WITH joined AS (
        SELECT
            m.id,
            w.year,
            from_json(m.genres, 'array<struct<id:int,name:string>>') AS genres_parsed      /*Convertir la chaîne JSON en array de structs */
        FROM movies m
        JOIN movies_with_year w
        ON m.id = w.id
        WHERE w.year != ''
    ),
    exploded AS (
        SELECT
            CAST(year AS INT) - (CAST(year AS INT) % 10) AS decade,                           /*Explosion de Genre*/
            explode(transform(genres_parsed, x -> x.name)) AS genre
        FROM joined
    ),
    genre_count AS (
        SELECT
            decade,
            genre,                                                  /* Pour chaque Decennie, on compte le  genre*/
            COUNT(*) AS nb_films
        FROM exploded
        GROUP BY decade, genre
    ),
    ranked AS (
        SELECT
            decade,
            genre,
            nb_films,                                              /* Avec le compte fait, on fait un classement du nombre de genre pour chauqe decenie*/
            ROW_NUMBER() OVER (PARTITION BY decade ORDER BY nb_films DESC) AS rank
        FROM genre_count
    )
    SELECT
        decade,
        genre,
        nb_films                                                     /* on retient le premier genre pour chaque decennie*/
    FROM ranked
    WHERE rank = 1
    ORDER BY decade
""")


classement.show()



# Question 1.1 f)
#Identification des Titres presents au moins en double + le nombre de leur presence + id

Detection_doublons = spark.sql("""
    SELECT
        original_title,
        COUNT(*) AS nb_versions,
        collect_list(id) AS ids_list
    FROM movies
    GROUP BY original_title
    HAVING COUNT(*) >= 2
    ORDER BY nb_versions DESC
""")
          
Detection_doublons.show(20, truncate=50)





