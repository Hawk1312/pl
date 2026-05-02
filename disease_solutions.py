DISEASE_SOLUTIONS = {
    "Pepper__bell___Bacterial_spot": (
        "Remove infected leaves, avoid overhead watering, use disease-free seed, "
        "rotate crops, and apply a copper-based bactericide as advised locally."
    ),
    "Pepper__bell___healthy": (
        "The plant appears healthy. Continue balanced watering, sunlight, "
        "field sanitation, and regular monitoring."
    ),
    "Potato___Early_blight": (
        "Remove infected leaves, rotate with non-host crops, avoid wet foliage, "
        "improve spacing, and use recommended fungicides such as chlorothalonil "
        "or mancozeb under local agriculture guidance."
    ),
    "Potato___Late_blight": (
        "Destroy badly infected plants, avoid overhead irrigation, improve airflow, "
        "and apply protective fungicides quickly. Late blight spreads fast, so "
        "isolate affected plants."
    ),
    "Potato___healthy": (
        "The plant appears healthy. Keep soil moisture steady, avoid leaf wetness, "
        "and scout weekly for early symptoms."
    ),
    "Tomato_Bacterial_spot": (
        "Remove infected leaves, avoid splashing water, disinfect tools, rotate "
        "crops, and use copper-based sprays where recommended."
    ),
    "Tomato_Early_blight": (
        "Prune lower infected leaves, mulch soil, avoid overhead watering, rotate "
        "crops, and apply recommended fungicide if disease pressure is high."
    ),
    "Tomato_healthy": (
        "The plant appears healthy. Maintain proper watering, nutrition, airflow, "
        "and regular pest checks."
    ),
    "Tomato_Late_blight": (
        "Remove infected plants or leaves quickly, avoid wet leaves, increase "
        "spacing, and use locally recommended fungicides. Do not compost infected "
        "material."
    ),
    "Tomato_Leaf_Mold": (
        "Improve greenhouse or field ventilation, reduce humidity, avoid wetting "
        "leaves, remove infected foliage, and use labeled fungicide if needed."
    ),
    "Tomato_Septoria_leaf_spot": (
        "Remove lower infected leaves, mulch to stop soil splash, avoid overhead "
        "watering, rotate crops, and use protective fungicide when required."
    ),
    "Tomato_Spider_mites_Two_spotted_spider_mite": (
        "Spray leaves with water to reduce mites, remove heavily affected leaves, "
        "encourage beneficial insects, and use miticide or insecticidal soap if "
        "infestation is severe."
    ),
    "Tomato__Target_Spot": (
        "Remove infected leaves, improve airflow, avoid overhead irrigation, rotate "
        "crops, and apply recommended fungicide during humid weather."
    ),
    "Tomato__Tomato_mosaic_virus": (
        "Remove infected plants, disinfect hands and tools, control tobacco handling "
        "near plants, and grow resistant varieties. There is no direct cure for "
        "infected plants."
    ),
    "Tomato__Tomato_YellowLeaf__Curl_Virus": (
        "Remove infected plants, control whiteflies with traps or recommended "
        "insecticides, remove weeds, and use resistant varieties. Viral infections "
        "cannot be cured once established."
    ),
}


def get_solution(class_name: str) -> str:
    if class_name in DISEASE_SOLUTIONS:
        return DISEASE_SOLUTIONS[class_name]

    if "healthy" in class_name.lower():
        return (
            "The plant appears healthy. Continue good watering, nutrition, "
            "sanitation, and regular monitoring."
        )

    return (
        "Remove infected leaves, isolate badly affected plants, avoid overhead "
        "watering, improve airflow, and ask a local agriculture expert for the "
        "correct pesticide or fungicide."
    )
