#!/usr/bin/env python3
"""
Generate 500 Unique Haiku Author Personas
Creates diverse author profiles with locations, education, and experience
"""

import json
import random
from datetime import datetime


# Leading Poetry Education Institutions and Programs
POETRY_EDUCATORS = [
    # US Universities - MFA Programs
    {"name": "Iowa Writers' Workshop", "location": "Iowa City, Iowa, USA", "type": "MFA"},
    {"name": "Stanford University", "location": "Stanford, California, USA", "type": "MFA"},
    {"name": "University of Michigan", "location": "Ann Arbor, Michigan, USA", "type": "MFA"},
    {"name": "Columbia University", "location": "New York, New York, USA", "type": "MFA"},
    {"name": "New York University", "location": "New York, New York, USA", "type": "MFA"},
    {"name": "University of California, Irvine", "location": "Irvine, California, USA", "type": "MFA"},
    {"name": "Boston University", "location": "Boston, Massachusetts, USA", "type": "MFA"},
    {"name": "University of Virginia", "location": "Charlottesville, Virginia, USA", "type": "MFA"},
    {"name": "Johns Hopkins University", "location": "Baltimore, Maryland, USA", "type": "MA"},
    {"name": "University of Texas, Austin", "location": "Austin, Texas, USA", "type": "MFA"},

    # International Universities
    {"name": "University of East Anglia", "location": "Norwich, England, UK", "type": "MA"},
    {"name": "University of Oxford", "location": "Oxford, England, UK", "type": "MSt"},
    {"name": "University of Edinburgh", "location": "Edinburgh, Scotland, UK", "type": "MLitt"},
    {"name": "University of Manchester", "location": "Manchester, England, UK", "type": "MA"},
    {"name": "University of Toronto", "location": "Toronto, Ontario, Canada", "type": "MFA"},
    {"name": "University of Melbourne", "location": "Melbourne, Victoria, Australia", "type": "MFA"},
    {"name": "University of Sydney", "location": "Sydney, New South Wales, Australia", "type": "MFA"},

    # Japanese Haiku Institutions
    {"name": "Tokyo University", "location": "Tokyo, Japan", "type": "Traditional Studies"},
    {"name": "Kyoto University", "location": "Kyoto, Japan", "type": "Traditional Studies"},
    {"name": "Waseda University", "location": "Tokyo, Japan", "type": "Literature"},

    # Poetry Centers and Workshops
    {"name": "Bread Loaf Writers' Conference", "location": "Vermont, USA", "type": "Workshop"},
    {"name": "Sewanee Writers' Conference", "location": "Tennessee, USA", "type": "Workshop"},
    {"name": "Fine Arts Work Center", "location": "Provincetown, Massachusetts, USA", "type": "Fellowship"},
    {"name": "Cave Canem", "location": "Various locations, USA", "type": "Fellowship"},
    {"name": "Kundiman", "location": "Various locations, USA", "type": "Workshop"},
    {"name": "Naropa University", "location": "Boulder, Colorado, USA", "type": "MFA"},
    {"name": "Poetry Society of America", "location": "New York, USA", "type": "Workshops"},
    {"name": "Academy of American Poets", "location": "New York, USA", "type": "Seminars"},

    # International Poetry Centers
    {"name": "Arvon Foundation", "location": "Various, UK", "type": "Workshop"},
    {"name": "Poetry School London", "location": "London, England, UK", "type": "Courses"},
    {"name": "Banff Centre", "location": "Banff, Alberta, Canada", "type": "Residency"},

    # Mentorship and Independent Study
    {"name": "Independent study with established haiku masters", "location": "Various", "type": "Mentorship"},
    {"name": "Haiku Society of America", "location": "Various, USA", "type": "Community"},
    {"name": "British Haiku Society", "location": "UK", "type": "Community"},
    {"name": "Self-taught through extensive reading", "location": "Various", "type": "Autodidact"},
]

# Global Locations for diverse backgrounds
GLOBAL_LOCATIONS = [
    # North America
    "New York, USA", "Los Angeles, USA", "Chicago, USA", "San Francisco, USA",
    "Seattle, USA", "Portland, USA", "Austin, USA", "Denver, USA",
    "Boston, USA", "Philadelphia, USA", "Miami, USA", "New Orleans, USA",
    "Toronto, Canada", "Vancouver, Canada", "Montreal, Canada",
    "Mexico City, Mexico", "Guadalajara, Mexico",

    # Europe
    "London, UK", "Edinburgh, UK", "Manchester, UK", "Dublin, Ireland",
    "Paris, France", "Lyon, France", "Berlin, Germany", "Munich, Germany",
    "Rome, Italy", "Florence, Italy", "Barcelona, Spain", "Madrid, Spain",
    "Amsterdam, Netherlands", "Copenhagen, Denmark", "Stockholm, Sweden",
    "Oslo, Norway", "Helsinki, Finland", "Reykjavik, Iceland",
    "Prague, Czech Republic", "Vienna, Austria", "Zurich, Switzerland",
    "Athens, Greece", "Lisbon, Portugal", "Brussels, Belgium",

    # Asia
    "Tokyo, Japan", "Kyoto, Japan", "Osaka, Japan", "Hiroshima, Japan",
    "Seoul, South Korea", "Busan, South Korea",
    "Beijing, China", "Shanghai, China", "Hong Kong, China",
    "Taipei, Taiwan", "Bangkok, Thailand", "Singapore",
    "Hanoi, Vietnam", "Ho Chi Minh City, Vietnam",
    "Mumbai, India", "Delhi, India", "Bangalore, India", "Kolkata, India",
    "Kathmandu, Nepal", "Dhaka, Bangladesh",
    "Jakarta, Indonesia", "Manila, Philippines",

    # Oceania
    "Sydney, Australia", "Melbourne, Australia", "Brisbane, Australia",
    "Wellington, New Zealand", "Auckland, New Zealand",

    # Middle East
    "Istanbul, Turkey", "Tel Aviv, Israel", "Jerusalem, Israel",
    "Dubai, UAE", "Beirut, Lebanon", "Cairo, Egypt",

    # South America
    "Buenos Aires, Argentina", "São Paulo, Brazil", "Rio de Janeiro, Brazil",
    "Santiago, Chile", "Lima, Peru", "Bogotá, Colombia",
    "Montevideo, Uruguay", "Quito, Ecuador",

    # Africa
    "Cape Town, South Africa", "Johannesburg, South Africa",
    "Lagos, Nigeria", "Nairobi, Kenya", "Accra, Ghana",
    "Addis Ababa, Ethiopia", "Casablanca, Morocco",

    # Rural/Small Town (various countries)
    "Rural Vermont, USA", "Small town Maine, USA", "Rural Scotland, UK",
    "Village in Tuscany, Italy", "Countryside Japan", "Rural Ireland",
    "Small town Canada", "Coastal Norway", "Mountain village Switzerland",
    "Rural New Zealand", "Outback Australia", "Rural India",
]

# Language mapping for locations
LOCATION_LANGUAGES = {
    # North America
    "USA": "English",
    "Canada": ["English", "French"],
    "Mexico": "Spanish",

    # Europe
    "UK": "English",
    "Ireland": "English",
    "France": "French",
    "Germany": "German",
    "Italy": "Italian",
    "Spain": "Spanish",
    "Netherlands": "Dutch",
    "Denmark": "Danish",
    "Sweden": "Swedish",
    "Norway": "Norwegian",
    "Finland": "Finnish",
    "Iceland": "Icelandic",
    "Czech Republic": "Czech",
    "Austria": "German",
    "Switzerland": ["German", "French", "Italian"],
    "Greece": "Greek",
    "Portugal": "Portuguese",
    "Belgium": ["Dutch", "French"],

    # Asia
    "Japan": "Japanese",
    "South Korea": "Korean",
    "Korea": "Korean",
    "China": "Mandarin",
    "Hong Kong": ["Cantonese", "English"],
    "Taiwan": "Mandarin",
    "Thailand": "Thai",
    "Singapore": ["English", "Mandarin"],
    "Vietnam": "Vietnamese",
    "India": ["Hindi", "English"],
    "Nepal": "Nepali",
    "Bangladesh": "Bengali",
    "Indonesia": "Indonesian",
    "Philippines": ["Filipino", "English"],

    # Oceania
    "Australia": "English",
    "New Zealand": "English",

    # Middle East
    "Turkey": "Turkish",
    "Israel": ["Hebrew", "Arabic"],
    "UAE": "Arabic",
    "Lebanon": ["Arabic", "French"],
    "Egypt": "Arabic",

    # South America
    "Argentina": "Spanish",
    "Brazil": "Portuguese",
    "Chile": "Spanish",
    "Peru": "Spanish",
    "Colombia": "Spanish",
    "Uruguay": "Spanish",
    "Ecuador": "Spanish",

    # Africa
    "South Africa": ["English", "Afrikaans"],
    "Nigeria": ["English", "Yoruba", "Igbo"],
    "Kenya": ["Swahili", "English"],
    "Ghana": "English",
    "Ethiopia": "Amharic",
    "Morocco": ["Arabic", "French"],
}


def get_language_from_location(location: str) -> list:
    """Extract primary language(s) from a location string"""
    languages = []

    # Check each country/region in the mapping
    for region, lang in LOCATION_LANGUAGES.items():
        if region in location:
            if isinstance(lang, list):
                languages.extend(lang)
            else:
                languages.append(lang)
            break

    # If no match found, default to English (international lingua franca)
    if not languages:
        languages.append("English")

    return languages


def calculate_languages(childhood, occupation, education, current):
    """Calculate language proficiency based on all locations lived"""
    languages = set()

    # Native language from childhood
    native = get_language_from_location(childhood)
    languages.update(native)

    # Languages from other locations
    languages.update(get_language_from_location(occupation))
    languages.update(get_language_from_location(education))
    languages.update(get_language_from_location(current))

    # Always include English as it's the international language of poetry education
    languages.add("English")

    # Sort for consistency
    languages_list = sorted(list(languages))

    # Determine proficiency
    proficiency = {
        "native": native,
        "fluent": languages_list,
        "total_languages": len(languages_list)
    }

    return proficiency


# Name generation by cultural region
# Using gender-neutral or culturally appropriate given names + surnames
NAMES_BY_REGION = {
    "USA": {
        "given": ["Taylor", "Jordan", "Morgan", "Casey", "Riley", "Alex", "Jamie", "Robin",
                 "Quinn", "Dakota", "Sage", "River", "Skyler", "Rowan", "Blake", "Harper",
                 "Avery", "Parker", "Emerson", "Finley", "Peyton", "Cameron", "Logan", "Drew"],
        "surnames": ["Chen", "Martinez", "Williams", "Johnson", "Brown", "Davis", "Miller", "Garcia",
                    "Rodriguez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "White",
                    "Harris", "Clark", "Lewis", "Walker", "Hall", "Young", "King", "Wright"]
    },
    "Canada": {
        "given": ["Alex", "Sam", "Jordan", "Taylor", "Morgan", "Riley", "Cameron", "Quinn",
                 "River", "Sky", "Harper", "Parker", "Robin", "Sage", "Blair", "Charlie"],
        "surnames": ["Leblanc", "Roy", "Gagnon", "Wong", "Singh", "MacDonald", "Campbell", "Stewart",
                    "Smith", "Brown", "Wilson", "Lee", "Kim", "Nguyen", "Patel", "Cohen"]
    },
    "Mexico": {
        "given": ["Guadalupe", "Andrea", "Ariel", "Azul", "Cruz", "Mar", "Paz", "Sol"],
        "surnames": ["García", "Rodríguez", "Hernández", "López", "Martínez", "González", "Pérez", "Sánchez",
                    "Ramírez", "Torres", "Flores", "Rivera", "Gómez", "Díaz", "Morales", "Jiménez"]
    },
    "UK": {
        "given": ["Alex", "Charlie", "Jamie", "Morgan", "Riley", "Sam", "Taylor", "Robin",
                 "Ash", "Kit", "Sage", "River", "Quinn", "Jordan", "Rowan", "Blair"],
        "surnames": ["Smith", "Jones", "Williams", "Brown", "Taylor", "Davies", "Wilson", "Evans",
                    "Thomas", "Roberts", "Johnson", "Walker", "White", "Hughes", "Green", "Lewis"]
    },
    "Ireland": {
        "given": ["Aoife", "Cian", "Ronan", "Saoirse", "Quinn", "Riley", "Ryan", "Morgan"],
        "surnames": ["Murphy", "Kelly", "O'Brien", "Ryan", "O'Sullivan", "Walsh", "Smith", "O'Connor",
                    "McCarthy", "Gallagher", "Doherty", "Kennedy", "Lynch", "Murray", "Quinn", "Moore"]
    },
    "France": {
        "given": ["Alex", "Dominique", "Claude", "Sacha", "Camille", "Lou", "Charlie", "Morgan"],
        "surnames": ["Martin", "Bernard", "Dubois", "Thomas", "Robert", "Richard", "Petit", "Durand",
                    "Leroy", "Moreau", "Simon", "Laurent", "Lefebvre", "Michel", "Garcia", "David"]
    },
    "Germany": {
        "given": ["Alex", "Kim", "Robin", "Sam", "Charlie", "Lou", "Noa", "Jona"],
        "surnames": ["Müller", "Schmidt", "Schneider", "Fischer", "Weber", "Meyer", "Wagner", "Becker",
                    "Schulz", "Hoffmann", "Koch", "Richter", "Klein", "Wolf", "Schröder", "Neumann"]
    },
    "Italy": {
        "given": ["Andrea", "Simone", "Nicola", "Alex", "Celeste", "Fiore", "Luca", "Sole"],
        "surnames": ["Rossi", "Russo", "Ferrari", "Esposito", "Bianchi", "Romano", "Colombo", "Ricci",
                    "Marino", "Greco", "Bruno", "Gallo", "Conti", "DeLuca", "Costa", "Giordano"]
    },
    "Spain": {
        "given": ["Alex", "Andrea", "Ariel", "Cruz", "Mar", "Paz", "Sacha", "Sol"],
        "surnames": ["García", "Rodríguez", "González", "Fernández", "López", "Martínez", "Sánchez", "Pérez",
                    "Gómez", "Martín", "Jiménez", "Ruiz", "Hernández", "Díaz", "Moreno", "Muñoz"]
    },
    "Japan": {
        "given": ["Akira", "Hikari", "Kaori", "Makoto", "Nao", "Ren", "Sora", "Yuki",
                 "Haruki", "Aoi", "Michi", "Kei", "Ryo", "Shun", "Jun", "Tomo"],
        "surnames": ["Sato", "Suzuki", "Takahashi", "Tanaka", "Watanabe", "Ito", "Yamamoto", "Nakamura",
                    "Kobayashi", "Kato", "Yoshida", "Yamada", "Sasaki", "Yamaguchi", "Saito", "Matsumoto"]
    },
    "Korea": {
        "given": ["Ji", "Min", "Seo", "Hyun", "Young", "Jun", "Hae", "Sun"],
        "surnames": ["Kim", "Lee", "Park", "Choi", "Jung", "Kang", "Cho", "Yoon",
                    "Jang", "Lim", "Han", "Oh", "Seo", "Shin", "Kwon", "Song"]
    },
    "China": {
        "given": ["Wei", "Lin", "Ming", "Ying", "Chen", "Hui", "Xin", "Yu",
                 "Jing", "Li", "Mei", "Xia", "Jun", "Wen", "Yun", "Qing"],
        "surnames": ["Wang", "Li", "Zhang", "Liu", "Chen", "Yang", "Huang", "Zhao",
                    "Wu", "Zhou", "Xu", "Sun", "Ma", "Zhu", "Hu", "Guo"]
    },
    "India": {
        "given": ["Arjun", "Ananya", "Rohan", "Priya", "Aditya", "Kavya", "Nikhil", "Shreya",
                 "Arnav", "Diya", "Vivaan", "Aarav", "Saanvi", "Riya", "Krishna", "Ishaan"],
        "surnames": ["Sharma", "Kumar", "Singh", "Patel", "Gupta", "Reddy", "Iyer", "Nair",
                    "Desai", "Mehta", "Shah", "Chopra", "Malhotra", "Kapoor", "Khan", "Das"]
    },
    "Australia": {
        "given": ["Alex", "Sam", "Jordan", "Taylor", "Morgan", "Riley", "Casey", "Charlie",
                 "Jamie", "Ash", "River", "Sky", "Harper", "Quinn", "Blake", "Parker"],
        "surnames": ["Smith", "Jones", "Williams", "Brown", "Wilson", "Taylor", "Johnson", "White",
                    "Martin", "Anderson", "Thompson", "Nguyen", "Lee", "Walker", "Harris", "Patel"]
    },
    "Brazil": {
        "given": ["Alex", "Morgan", "Ariel", "Cruz", "Sol", "Davi", "Luca", "Yuri"],
        "surnames": ["Silva", "Santos", "Oliveira", "Souza", "Rodrigues", "Ferreira", "Alves", "Pereira",
                    "Lima", "Gomes", "Costa", "Ribeiro", "Martins", "Carvalho", "Rocha", "Almeida"]
    },
    "Argentina": {
        "given": ["Alex", "Cruz", "Mar", "Sol", "Ariel", "Morgan", "Franco", "Valentín"],
        "surnames": ["González", "Rodríguez", "García", "Fernández", "López", "Martínez", "Pérez", "Sánchez",
                    "Romero", "Díaz", "Torres", "Álvarez", "Ruiz", "Gómez", "Hernández", "Vázquez"]
    },
    "default": {
        "given": ["Alex", "Sam", "Jordan", "Morgan", "Taylor", "Casey", "Riley", "Quinn",
                 "River", "Sky", "Sage", "Robin", "Parker", "Harper", "Blake", "Cameron"],
        "surnames": ["Smith", "Chen", "García", "Singh", "Kim", "Santos", "Müller", "Sato",
                    "Ali", "Novak", "Andersson", "Patel", "Kowalski", "Martin", "Brown", "Lee"]
    }
}

# Avoided names (famous haiku poets - to ensure uniqueness)
AVOIDED_NAMES = [
    "Basho", "Matsuo", "Buson", "Issa", "Kobayashi", "Shiki", "Masaoka",
    "Santoka", "Taneda", "Hosai", "Ozaki", "Seisensui", "Ogiwara",
    "Kerouac", "Snyder", "Ginsberg", "Corso", "Ferlinghetti",
    "Wright", "Sanchez", "Kaufman", "Rexroth", "Whalen",
    "Lowell", "Plath", "Bishop", "Pound", "Williams",
    "Haiku", "Hokku", "Renga", "Tanka"
]


def get_name_for_region(location: str, birth_year: int) -> dict:
    """Generate culturally appropriate name based on birth location"""

    # Determine region from location
    region = "default"
    for key in NAMES_BY_REGION.keys():
        if key in location:
            region = key
            break

    names = NAMES_BY_REGION.get(region, NAMES_BY_REGION["default"])

    # Select given name and surname
    given_name = random.choice(names["given"])
    surname = random.choice(names["surnames"])

    # Check against avoided names
    attempts = 0
    while (surname in AVOIDED_NAMES or given_name in AVOIDED_NAMES) and attempts < 10:
        given_name = random.choice(names["given"])
        surname = random.choice(names["surnames"])
        attempts += 1

    # Determine if name might have changed (more common for older people, certain cultures)
    name_changed = False
    previous_name = None

    age = 2026 - birth_year
    if age > 30 and random.random() < 0.3:  # 30% chance for those over 30
        name_changed = True
        # Previous surname could be from same or different cultural background
        previous_surnames = names["surnames"] + NAMES_BY_REGION["default"]["surnames"]
        previous_name = random.choice([s for s in previous_surnames if s != surname])

    return {
        "current_name": f"{given_name} {surname}",
        "given_name": given_name,
        "surname": surname,
        "name_changed": name_changed,
        "previous_surname": previous_name if name_changed else None,
        "birth_name": f"{given_name} {previous_name}" if name_changed else f"{given_name} {surname}"
    }


# Professional Occupations
OCCUPATIONS = [
    "Teacher", "Professor", "Librarian", "Bookseller",
    "Journalist", "Editor", "Translator", "Publisher",
    "Software Engineer", "Data Analyst", "Web Developer", "IT Consultant",
    "Doctor", "Nurse", "Therapist", "Social Worker",
    "Lawyer", "Paralegal", "Judge", "Legal Aid Worker",
    "Architect", "Engineer", "Urban Planner", "Designer",
    "Artist", "Musician", "Photographer", "Filmmaker",
    "Chef", "Restaurant Owner", "Food Critic", "Baker",
    "Farmer", "Gardener", "Landscape Architect", "Botanist",
    "Scientist", "Researcher", "Lab Technician", "Environmental Scientist",
    "Accountant", "Financial Analyst", "Banker", "Insurance Agent",
    "Marketing Manager", "Advertising Executive", "PR Specialist", "Brand Strategist",
    "Retail Manager", "Store Owner", "Sales Representative", "Buyer",
    "Construction Worker", "Electrician", "Plumber", "Carpenter",
    "Mechanic", "Driver", "Pilot", "Ship Captain",
    "Police Officer", "Firefighter", "Paramedic", "Military Veteran",
    "Social Activist", "Non-profit Director", "Community Organizer", "Volunteer Coordinator",
    "Religious Leader", "Counselor", "Life Coach", "Spiritual Guide",
    "Stay-at-home Parent", "Retired Professional", "Student", "Unemployed",
    "Freelance Writer", "Independent Consultant", "Entrepreneur", "Business Owner",
]


def generate_persona(persona_id):
    """Generate a single unique persona"""

    # Age between 18 and 100
    age = random.randint(18, 100)

    # Work experience (0-60 years, but not exceeding age-18)
    max_experience = min(60, age - 18)
    work_experience = random.randint(0, max_experience)

    # Education
    education = random.choice(POETRY_EDUCATORS)

    # Locations
    childhood_location = random.choice(GLOBAL_LOCATIONS)
    occupation_location = random.choice(GLOBAL_LOCATIONS)
    poetry_education_location = education["location"]
    current_location = random.choice(GLOBAL_LOCATIONS)

    # Occupation (may be retired if old enough)
    if age >= 65 and random.random() > 0.5:
        occupation = "Retired " + random.choice(OCCUPATIONS)
    else:
        occupation = random.choice(OCCUPATIONS)

    # Generate a brief characteristic
    characteristics = [
        "writes primarily at dawn",
        "focuses on urban landscapes",
        "draws inspiration from nature walks",
        "incorporates cultural heritage",
        "explores themes of displacement",
        "writes bilingually",
        "specializes in seasonal kigo",
        "experiments with form",
        "documents daily observations",
        "blends traditional and modern",
        "influenced by jazz rhythms",
        "writes from personal trauma",
        "focuses on environmental themes",
        "explores aging and mortality",
        "documents working-class life",
        "writes about immigration",
        "explores spiritual themes",
        "focuses on urban nature",
        "writes political haiku",
        "documents pandemic life",
    ]

    # Calculate languages based on all locations
    language_proficiency = calculate_languages(
        childhood_location,
        occupation_location,
        poetry_education_location,
        current_location
    )

    # Generate name based on birth location and year
    birth_year = 2026 - age
    name_info = get_name_for_region(childhood_location, birth_year)

    persona = {
        "id": persona_id,
        "name": name_info["current_name"],
        "name_details": {
            "given_name": name_info["given_name"],
            "surname": name_info["surname"],
            "birth_name": name_info["birth_name"],
            "name_changed": name_info["name_changed"],
            "previous_surname": name_info["previous_surname"]
        },
        "age": age,
        "birth_year": birth_year,
        "work_experience_years": work_experience,
        "occupation": occupation,
        "poetry_education": {
            "institution": education["name"],
            "location": poetry_education_location,
            "type": education["type"]
        },
        "locations": {
            "childhood": childhood_location,
            "occupation": occupation_location,
            "poetry_education": poetry_education_location,
            "current": current_location
        },
        "languages": language_proficiency,
        "characteristic": random.choice(characteristics),
        "active_since": 2026 - random.randint(1, max(1, min(40, age - 18)))
    }

    return persona


def generate_all_personas(count=500):
    """Generate all personas"""
    personas = []
    for i in range(1, count + 1):
        personas.append(generate_persona(i))
    return personas


def main():
    """Generate and save persona database"""
    print("Generating 500 haiku author personas...")

    personas = generate_all_personas(500)

    database = {
        "metadata": {
            "title": "Haiku Author Personas Database",
            "description": "500 unique author profiles with diverse backgrounds, locations, and experiences",
            "total_personas": len(personas),
            "generated": datetime.now().strftime("%Y-%m-%d"),
            "version": "1.0"
        },
        "personas": personas,
        "statistics": {
            "age_range": "18-100",
            "experience_range": "0-60 years",
            "total_education_institutions": len(POETRY_EDUCATORS),
            "total_locations": len(GLOBAL_LOCATIONS),
            "total_occupations": len(OCCUPATIONS)
        }
    }

    output_file = "haiku_personas.json"
    with open(output_file, 'w') as f:
        json.dump(database, f, indent=2)

    print(f"✓ Generated {len(personas)} personas")
    print(f"✓ Saved to: {output_file}")

    # Print sample personas
    print("\nSample personas:")
    for persona in random.sample(personas, 3):
        print(f"\nPersona #{persona['id']}:")
        print(f"  Age: {persona['age']}, Experience: {persona['work_experience_years']} years")
        print(f"  Occupation: {persona['occupation']}")
        print(f"  Education: {persona['poetry_education']['institution']}")
        print(f"  Current location: {persona['locations']['current']}")
        print(f"  Style: {persona['characteristic']}")


if __name__ == "__main__":
    main()
