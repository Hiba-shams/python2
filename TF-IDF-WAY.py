import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process  # Faster and lighter than fuzzywuzzy

# Load the dataset
cars_df = pd.read_csv("car_rental_data.csv")

# Ensure the 'features' column is parsed properly
cars_df['features'] = cars_df['features'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Combine relevant fields into a single text description
def combine_features(row):
    return (
        f"{row['brand']} {row['model']} {row['type']} {row['transmission']} " +
        " ".join(row['features'])
    )

cars_df['combined_features'] = cars_df.apply(combine_features, axis=1)

# Build vocabulary for typo correction
vocabulary = set()
for text in cars_df['combined_features']:
    for word in text.lower().split():
        vocabulary.add(word)

# Vectorize the combined features
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(cars_df['combined_features'])

# Function to correct typos in user query
def correct_typos(user_input):
    corrected_words = []
    for word in user_input.lower().split():
        match, score, _ = process.extractOne(word, vocabulary)
        if score > 80:  # Only correct if similarity is high
            corrected_words.append(match)
        else:
            corrected_words.append(word)  # Keep the original if no good match
    return " ".join(corrected_words)

# Search for cars based on user query
def search_cars(user_query, num_recommendations=3, min_price=0, max_price=100):
    corrected_query = correct_typos(user_query)
    query_vec = vectorizer.transform([corrected_query])
    sim_scores = list(enumerate(cosine_similarity(query_vec, tfidf_matrix)[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    for idx, score in sim_scores:
        car_row = cars_df.iloc[idx]
        if min_price <= car_row['price_per_day'] <= max_price:
            recommendations.append(car_row)
        if len(recommendations) >= num_recommendations:
            break

    return recommendations

# Interactive recommendation system
def start_recommendation_system():
    while True:
        print("\n🚗 Welcome to the Smart Car Rental Recommendation System!")
        print("Example Inputs: 'Toyota SUV automatic GPS', 'BMW sedan leather seats', or 'manual SUV sunroof'")
        user_input = input("Enter your car preference (or type 'exit' to quit): ").strip()

        if user_input.lower() == 'exit':
            print("👋 Goodbye! Thanks for using the system.")
            break

        try:
            min_price = int(input("Enter minimum price per day ($): ").strip())
            max_price = int(input("Enter maximum price per day ($): ").strip())
        except ValueError:
            print("❌ Invalid price input. Please enter numbers only.")
            continue

        recommendations = search_cars(
            user_query=user_input,
            num_recommendations=3,
            min_price=min_price,
            max_price=max_price
        )

        if recommendations:
            print("\n🔎 Top Recommended Cars:")
            for car in recommendations:
                print(f"- {car['brand']} {car['model']} ({car['type']}, {car['transmission']}, ${car['price_per_day']}/day) Features: {', '.join(car['features'])}")
        else:
            print("❌ No recommendations found for the given preferences and price range.")

# Run the system
if __name__ == "__main__":
    start_recommendation_system()
