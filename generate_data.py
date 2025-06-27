import pandas as pd
import random

# Seed for reproducibility
random.seed(42)

# Step 1: Original car rental dataset
car_data = {
    "car_id": ["C001", "C002", "C003", "C004", "C005", "C006", "C007", "C008", "C009", "C010"],
    "brand": ["Toyota", "Ford", "BMW", "Honda", "Hyundai", "Chevrolet", "Nissan", "Kia", "Mercedes", "Jeep"],
    "model": ["RAV4", "Escape", "3 Series", "Civic", "Tucson", "Malibu", "Altima", "Sportage", "C Class", "Wrangler"],
    "type": ["SUV", "SUV", "Sedan", "Sedan", "SUV", "Sedan", "Sedan", "SUV", "Sedan", "SUV"],
    "price_per_day": [65, 70, 85, 60, 68, 62, 64, 66, 90, 75],
    "transmission": ["automatic", "automatic", "automatic", "manual", "automatic", "automatic", "automatic", "automatic", "automatic", "manual"],
    "features": [
        ["GPS", "Bluetooth"],
        ["GPS", "Sunroof"],
        ["GPS", "Leather Seats"],
        ["Bluetooth", "Backup Camera"],
        ["GPS", "Backup Camera"],
        ["Sunroof", "Bluetooth"],
        ["GPS", "Bluetooth", "Heated Seats"],
        ["Bluetooth", "Backup Camera"],
        ["Leather Seats", "Backup Camera"],
        ["4WD", "GPS", "Bluetooth"]
    ],
    "mileage": [30000, 45000, 25000, 40000, 35000, 47000, 42000, 38000, 27000, 50000],
    "year": [2018, 2017, 2019, 2016, 2018, 2017, 2018, 2017, 2019, 2016],
    "rating": [4.5, 4.3, 4.7, 4.0, 4.2, 4.1, 4.4, 4.0, 4.6, 4.3]
}

# Load into DataFrame
cars_df = pd.DataFrame(car_data)

# Step 2: Function to generate random but realistic data
def generate_random_data(num_new_cars):
    brand_model_map = {
        "Toyota": ["RAV4", "Corolla", "Camry", "Highlander"],
        "Ford": ["Escape", "Focus", "Explorer"],
        "BMW": ["3 Series", "5 Series", "X3", "X5"],
        "Honda": ["Civic", "Accord", "CR-V"],
        "Hyundai": ["Tucson", "Elantra", "Santa Fe"],
        "Chevrolet": ["Malibu", "Equinox", "Traverse"],
        "Nissan": ["Altima", "Sentra", "Rogue"],
        "Kia": ["Sportage", "Sorento", "Optima"],
        "Mercedes": ["C Class", "E Class", "GLC"],
        "Jeep": ["Wrangler", "Cherokee", "Compass"]
    }
    
    types = ["SUV", "Sedan", "Truck", "Convertible"]
    transmissions = ["automatic", "manual"]
    features_pool = [
        "GPS", "Bluetooth", "Leather Seats", "Sunroof", 
        "Backup Camera", "Heated Seats", "4WD", "Parking Sensors"
    ]
    
    new_cars = []
    start_id = len(cars_df) + 1

    for i in range(num_new_cars):
        brand = random.choice(list(brand_model_map.keys()))
        model = random.choice(brand_model_map[brand])
        car_type = random.choice(types)
        transmission = random.choice(transmissions)
        
        # Price based on brand and type
        base_price = random.randint(50, 70)
        if brand in ["BMW", "Mercedes", "Jeep"]:
            base_price += random.randint(20, 40)  # Luxury brands
        
        if car_type == "Truck":
            base_price += 10  # Trucks slightly higher
        
        # Random features (2 to 5)
        features = random.sample(features_pool, random.randint(2, 5))
        
        mileage = random.randint(10000, 100000)  # km driven
        year = random.randint(2015, 2023)
        rating = round(random.uniform(3.5, 5.0), 1)  # Customer ratings

        car_id = f"C{str(start_id + i).zfill(3)}"
        
        new_cars.append({
            "car_id": car_id,
            "brand": brand,
            "model": model,
            "type": car_type,
            "price_per_day": base_price,
            "transmission": transmission,
            "features": features,
            "mileage": mileage,
            "year": year,
            "rating": rating
        })

    return pd.DataFrame(new_cars)

# Step 3: Generate new cars and combine
new_cars_df = generate_random_data(5000)
cars_df = pd.concat([cars_df, new_cars_df], ignore_index=True)

# Step 4: Save the dataset
cars_df.to_csv("car_rental_data.csv", index=False)
print("✅ dataset saved as 'car_rental_data.csv'")
