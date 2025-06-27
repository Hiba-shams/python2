import random
import json
import nlpaug.augmenter.word as naw

# Optional: Initialize NLPAug for question paraphrasing
paraphraser = naw.SynonymAug(aug_src='wordnet')

# Expanded topics and questions
topics = {
    "rental process": [
        "What documents are required to rent a car?",
        "Can I rent a car without a credit card?",
        "How long does it take to rent a car?",
        "Is international driving license accepted?"
    ],
    "insurance": [
        "What insurance options do I have when renting a car?",
        "Does rental include third-party insurance?",
        "Do I need to buy additional insurance?",
        "Will I be covered in case of an accident?"
    ],
    "pricing": [
        "How are car rental prices calculated?",
        "Are there any hidden fees in car rentals?",
        "Do prices vary depending on the rental duration?",
        "Is the price different for weekend rentals?"
    ],
    "age requirements": [
        "What is the minimum age to rent a car?",
        "Can a 19-year-old rent a car?",
        "Are there age-based restrictions for luxury cars?",
        "Are there extra charges for young drivers?"
    ],
    "pickup & dropoff": [
        "Can I return the car to a different location?",
        "Is it possible to pick up a car at the airport?",
        "What happens if I return the car late?",
        "Do I need to be present when dropping off the car?"
    ],
    "vehicle options": [
        "Can I choose the exact car I want?",
        "Do you offer SUVs for rent?",
        "Are electric vehicles available?",
        "Can I rent a luxury or convertible car?"
    ],
    "late returns": [
        "What fees apply for late returns?",
        "Is there a grace period for late returns?",
        "How is a late return calculated?"
    ],
    "cancellation": [
        "Can I cancel my reservation?",
        "Are there cancellation fees?",
        "How long before pickup can I cancel?"
    ],
    "fuel policy": [
        "Do I have to return the car with a full tank?",
        "What are your fuel policy options?",
        "Is fuel included in the rental price?"
    ],
    "loyalty programs": [
        "Do you offer a loyalty program?",
        "What are the benefits of your loyalty program?",
        "How can I earn points when renting?"
    ],
    "damage handling": [
        "What happens if I damage the car?",
        "Do I have to pay for minor scratches?",
        "Is there a deductible for damages?"
    ]
}

# Answers (multiple paraphrased per topic for variation)
answers = {
    "rental process": [
        "You'll usually need a driver’s license and a credit card to rent a car.",
        "Valid ID, a driver’s license, and a payment method are generally required."
    ],
    "insurance": [
        "Basic coverage is often included, but you can purchase extra insurance for more protection.",
        "You can choose from various insurance options including collision damage waivers and liability coverage."
    ],
    "pricing": [
        "Rental cost depends on the vehicle, duration, and any extras like insurance or GPS.",
        "Prices include base fees and taxes, and may vary by location or time of year."
    ],
    "age requirements": [
        "Most companies rent to drivers 21 and older, but some require you to be 25 for premium cars.",
        "There may be surcharges for drivers under 25 years old."
    ],
    "pickup & dropoff": [
        "Airport pickups are common and sometimes incur a convenience fee.",
        "Returning to a different location is allowed but may cost extra."
    ],
    "vehicle options": [
        "Most agencies offer a range of cars, including SUVs, sedans, and electric vehicles.",
        "You can request a type of car, but specific models aren't always guaranteed."
    ],
    "late returns": [
        "Late returns usually result in hourly or daily penalties.",
        "Some companies offer a short grace period before charging extra."
    ],
    "cancellation": [
        "You can usually cancel your reservation, but fees may apply depending on how close to pickup time.",
        "Free cancellation may be available up to 24 hours before pickup."
    ],
    "fuel policy": [
        "Most rentals require you to return the car with a full tank.",
        "Prepaid fuel options are sometimes offered for convenience."
    ],
    "loyalty programs": [
        "Loyalty programs reward frequent renters with discounts, upgrades, or free days.",
        "You earn points each time you rent, which can be redeemed later."
    ],
    "damage handling": [
        "You may be charged for any new damage; insurance can reduce your liability.",
        "Minor scratches or dents may fall under wear-and-tear policies depending on the provider."
    ]
}

# Simulate follow-up dialogue (multi-turn interaction)
follow_ups = {
    "insurance": ["Is it mandatory to take extra insurance?", "Will my personal car insurance work abroad?"],
    "pickup & dropoff": ["Can someone else pick up the car on my behalf?", "What documents are checked at pickup?"],
    "fuel policy": ["Can I pay for fuel in advance?", "What happens if I return the car without refueling?"],
    "damage handling": ["Can I report damage through an app?", "What if the damage is not my fault?"]
}

# Generate dataset
def generate_dataset(num_samples=1000, augment=False):
    dataset = []
    topic_keys = list(topics.keys())

    for _ in range(num_samples):
        topic = random.choice(topic_keys)
        base_question = random.choice(topics[topic])
        base_answer = random.choice(answers[topic])

        # Optional: Paraphrase question
        question = paraphraser.augment(base_question)[0] if augment else base_question

        # Base Q&A
        sample = {
            "prompt": f"Question: {question}\nAnswer:",
            "response": base_answer
        }
        dataset.append(sample)

        # Add follow-up if available
        if topic in follow_ups and random.random() < 0.3:
            follow_q = random.choice(follow_ups[topic])
            follow_a = random.choice(answers[topic])
            sample_follow = {
                "prompt": f"Follow-up: {follow_q}\nAnswer:",
                "response": follow_a
            }
            dataset.append(sample_follow)

    return dataset

# Generate and save to file
dataset = generate_dataset(1000, augment=True)

with open("enhanced_car_rental_dataset.jsonl", "w", encoding="utf-8") as f:
    for item in dataset:
        f.write(json.dumps(item) + "\n")

print("✅ Generated enhanced_car_rental_dataset.jsonl with augmented Q&A and multi-turn samples.")
