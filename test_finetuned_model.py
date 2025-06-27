import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import os

# Load the fine-tuned model and tokenizer
MODEL_PATH = "./car-rental-finetuned"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model path '{MODEL_PATH}' not found.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

# Initialize the generation pipeline
generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
)

# Define a categorized list of prompts for testing
prompts = [
    # 🔹 Rental Process
    "What documents are required to rent a car?",

    # 🔹 Insurance
    "What insurance options do I have when renting a car?",

    # 🔹 Age Requirements
    "Are there age-based restrictions for luxury cars?",
    "Are there extra charges for young drivers?",

    # 🔹 Pickup & Dropoff
    "Can I return the car to a different location?",
    "Is it possible to pick up a car at the airport?",

    # 🔹 Vehicle Options
    "Are electric vehicles available?",

    # 🔹 Late Returns
    "What fees apply for late returns?",
    "Is there a grace period for late returns?",
    "How is a late return calculated?",

    # 🔹 Cancellation
    "Can I cancel my reservation?",
    "How long before pickup can I cancel?",

    # 🔹 Fuel Policy
    "What are your fuel policy options?",
    "Is fuel included in the rental price?",
    "Can I pay for fuel in advance?",

    # 🔹 Loyalty Programs
    "Do you offer a loyalty program?",
    "What are the benefits of your loyalty program?",
    "How can I earn points when renting?",

    # 🔹 Damage Handling
    "What happens if I damage the car?",
    "Do I have to pay for minor scratches?",
    "Is there a deductible for damages?"
]


# Optional: set generation parameters
generation_kwargs = {
    "max_length": 128,
    "clean_up_tokenization_spaces": True,
    "do_sample": False,           # Set to True to add randomness
    "temperature": 0.7,           # Used when do_sample is True
    "top_p": 0.9,                 # Used when do_sample is True
    "num_return_sequences": 1
}

# Generate and print responses
print("🔍 Testing fine-tuned car rental model:\n")
results = []

for idx, prompt in enumerate(prompts, start=1):
    try:
        response = generator(prompt, **generation_kwargs)[0]["generated_text"]
        print(f"🔹 Prompt {idx}: {prompt}")
        print(f"✅ Response: {response}\n")
        results.append({"prompt": prompt, "response": response})
    except Exception as e:
        print(f"❌ Error generating response for prompt {idx}: {prompt}")
        print(f"   Reason: {e}\n")

# Optional: Save results to a file
with open("car_rental_test_results.jsonl", "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item) + "\n")

print("✅ All prompts processed and results saved to 'car_rental_test_results.jsonl'")
