
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz")
predictor.predict(
document="Besides its prominence in sports, Notre Dame is also a large, four-year, highly residential research University, and is consistently ranked among the top twenty universities in the United States  and as a major global university."
)
