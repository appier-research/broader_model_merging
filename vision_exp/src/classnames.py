from typing import List, Dict


MNIST_CLASSNAMES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
KMNIST_CLASSNAMES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
SVHN_CLASSNAMES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
EMNIST_MNIST_CLASSNAMES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
EMNIST_LETTERS_CLASSNAMES = [
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"
]

FASHION_MNIST_CLASSNAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

CIFAR10_CLASSNAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CIFAR100_CLASSNAMES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
    'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
    'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

STL10_CLASSNAMES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck"
]

DTD_CLASSNAMES = [
    'banded', 'blotchy', 'braided', 'bubbly', 'bumpy', 'chequered',
    'cobwebbed', 'cracked', 'crosshatched', 'crystalline', 'dotted',
    'fibrous', 'flecked', 'freckled', 'frilly', 'gauzy', 'grid',
    'grooved', 'honeycombed', 'interlaced', 'knitted', 'lacelike',
    'lined', 'marbled', 'matted', 'meshed', 'paisley', 'perforated',
    'pitted', 'pleated', 'polka-dotted', 'porous', 'potholed',
    'scaly', 'smeared', 'spiralled', 'sprinkled', 'stained',
    'stratified', 'striped', 'studded', 'swirly', 'veined',
    'waffled', 'woven', 'wrinkled', 'zigzagged'
]

EUROSAT_CLASSNAMES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake',
]

GTSRB_CLASSNAMES = [
    "speed limit 20", "speed limit 30", "speed limit 50", "speed limit 60",
    "speed limit 70", "speed limit 80", "restriction ends 80", "speed limit 100",
    "speed limit 120", "no overtaking", "no overtaking (trucks)",
    "priority at next intersection", "priority road", "give way", "stop",
    "no traffic both ways", "no trucks", "no entry", "danger", "bend left",
    "bend right", "bend", "uneven road", "slippery road", "road narrows",
    "construction", "traffic signal", "pedestrian crossing", "school crossing",
    "cycles crossing", "snow", "animals", "restriction ends", "go right",
    "go left", "go straight", "go right or straight", "go left or straight",
    "keep right", "keep left", "roundabout", "restriction ends (overtaking)",
    "restriction ends (overtaking (trucks))"
]

FER2013_CLASSNAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

RENDERED_SST2_CLASSNAMES = ["negative", "positive"]
PCAM_CLASSNAMES = ["no tumor", "tumor"]

SUN397_CLASSNAMES = [
    "abbey", "airplane cabin", "airport terminal", "alley", "amphitheater",
    "amusement arcade", "amusement park", "anechoic chamber", "apartment building outdoor",
    "apse indoor", "aquarium", "aqueduct", "arch", "archive", "arrival gate outdoor",
    "art gallery", "art school", "art studio", "assembly line", "athletic field outdoor",
    "atrium public", "attic", "auditorium", "auto factory", "badlands",
    "badminton court indoor", "baggage claim", "bakery shop", "balcony exterior", "balcony interior",
    "ball pit", "ballroom", "bamboo forest", "banquet hall", "bar",
    "barn", "barndoor", "baseball field", "basement", "basilica",
    "basketball court outdoor", "bathroom", "batters box", "bayou", "bazaar indoor",
    "bazaar outdoor", "beach", "beauty salon", "bedroom", "berth",
    "biology laboratory", "bistro indoor", "boardwalk", "boat deck", "boathouse",
    "bookstore", "booth indoor", "botanical garden", "bow window indoor", "bow window outdoor",
    "bowling alley", "boxing ring", "brewery indoor", "bridge", "building facade",
    "bullring", "burial chamber", "bus interior", "butchers shop", "butte",
    "cabin outdoor", "cafeteria", "campsite", "campus", "canal natural",
    "canal urban", "candy store", "canyon", "car interior backseat", "car interior frontseat",
    "carrousel", "casino indoor", "castle", "catacomb", "cathedral indoor",
    "cathedral outdoor", "cavern indoor", "cemetery", "chalet", "cheese factory",
    "chemistry lab", "chicken coop indoor", "chicken coop outdoor", "childs room", "church indoor",
    "church outdoor", "classroom", "clean room", "cliff", "cloister indoor",
    "closet", "clothing store", "coast", "cockpit", "coffee shop",
    "computer room", "conference center", "conference room", "construction site", "control room",
    "control tower outdoor", "corn field", "corral", "corridor", "cottage garden",
    "courthouse", "courtroom", "courtyard", "covered bridge exterior", "creek",
    "crevasse", "crosswalk", "cubicle office", "dam", "delicatessen",
    "dentists office", "desert sand", "desert vegetation", "diner indoor", "diner outdoor",
    "dinette home", "dinette vehicle", "dining car", "dining room", "discotheque",
    "dock", "doorway outdoor", "dorm room", "driveway", "driving range outdoor",
    "drugstore", "electrical substation", "elevator door", "elevator interior", "elevator shaft",
    "engine room", "escalator indoor", "excavation", "factory indoor", "fairway",
    "fastfood restaurant", "field cultivated", "field wild", "fire escape", "fire station",
    "firing range indoor", "fishpond", "florist shop indoor", "food court", "forest broadleaf",
    "forest needleleaf", "forest path", "forest road", "formal garden", "fountain",
    "galley", "game room", "garage indoor", "garbage dump", "gas station",
    "gazebo exterior", "general store indoor", "general store outdoor", "gift shop", "golf course",
    "greenhouse indoor", "greenhouse outdoor", "gymnasium indoor", "hangar indoor", "hangar outdoor",
    "harbor", "hayfield", "heliport", "herb garden", "highway",
    "hill", "home office", "hospital", "hospital room", "hot spring",
    "hot tub outdoor", "hotel outdoor", "hotel room", "house", "hunting lodge outdoor",
    "ice cream parlor", "ice floe", "ice shelf", "ice skating rink indoor", "ice skating rink outdoor",
    "iceberg", "igloo", "industrial area", "inn outdoor", "islet",
    "jacuzzi indoor", "jail cell", "jail indoor", "jewelry shop", "kasbah",
    "kennel indoor", "kennel outdoor", "kindergarden classroom", "kitchen", "kitchenette",
    "labyrinth outdoor", "lake natural", "landfill", "landing deck", "laundromat",
    "lecture room", "library indoor", "library outdoor", "lido deck outdoor", "lift bridge",
    "lighthouse", "limousine interior", "living room", "lobby", "lock chamber",
    "locker room", "mansion", "manufactured home", "market indoor", "market outdoor",
    "marsh", "martial arts gym", "mausoleum", "medina", "moat water",
    "monastery outdoor", "mosque indoor", "mosque outdoor", "motel", "mountain",
    "mountain snowy", "movie theater indoor", "museum indoor", "music store", "music studio",
    "nuclear power plant outdoor", "nursery", "oast house", "observatory outdoor", "ocean",
    "office", "office building", "oil refinery outdoor", "oilrig", "operating room",
    "orchard", "outhouse outdoor", "pagoda", "palace", "pantry",
    "park", "parking garage indoor", "parking garage outdoor", "parking lot", "parlor",
    "pasture", "patio", "pavilion", "pharmacy", "phone booth",
    "physics laboratory", "picnic area", "pilothouse indoor", "planetarium outdoor", "playground",
    "playroom", "plaza", "podium indoor", "podium outdoor", "pond",
    "poolroom establishment", "poolroom home", "power plant outdoor", "promenade deck", "pub indoor",
    "pulpit", "putting green", "racecourse", "raceway", "raft",
    "railroad track", "rainforest", "reception", "recreation room", "residential neighborhood",
    "restaurant", "restaurant kitchen", "restaurant patio", "rice paddy", "riding arena",
    "river", "rock arch", "rope bridge", "ruin", "runway",
    "sandbar", "sandbox", "sauna", "schoolhouse", "sea cliff",
    "server room", "shed", "shoe shop", "shopfront", "shopping mall indoor",
    "shower", "skatepark", "ski lodge", "ski resort", "ski slope",
    "sky", "skyscraper", "slum", "snowfield", "squash court",
    "stable", "stadium baseball", "stadium football", "stage indoor", "staircase",
    "street", "subway interior", "subway station platform", "supermarket", "sushi bar",
    "swamp", "swimming pool indoor", "swimming pool outdoor", "synagogue indoor", "synagogue outdoor",
    "television studio", "temple east asia", "temple south asia", "tennis court indoor", "tennis court outdoor",
    "tent outdoor", "theater indoor procenium", "theater indoor seats", "thriftshop", "throne room",
    "ticket booth", "toll plaza", "topiary garden", "tower", "toyshop",
    "track outdoor", "train railway", "train station platform", "tree farm", "tree house",
    "trench", "underwater coral reef", "utility room", "valley", "van interior",
    "vegetable garden", "veranda", "veterinarians office", "viaduct", "videostore",
    "village", "vineyard", "volcano", "volleyball court indoor", "volleyball court outdoor",
    "waiting room", "warehouse indoor", "water tower", "waterfall block", "waterfall fan",
    "waterfall plunge", "watering hole", "wave", "wet bar", "wheat field",
    "wind farm", "windmill", "wine cellar barrel storage", "wine cellar bottle storage", "wrestling ring indoor",
    "yard", "youth hostel"
]

STANFORD_CARS_CLASSNAMES = [
    "AM General Hummer SUV 2000", "Acura RL Sedan 2012", "Acura TL Sedan 2012",
    "Acura TL Type-S 2008", "Acura TSX Sedan 2012", "Acura Integra Type R 2001",
    "Acura ZDX Hatchback 2012", "Aston Martin V8 Vantage Convertible 2012",
    "Aston Martin V8 Vantage Coupe 2012", "Aston Martin Virage Convertible 2012",
    "Aston Martin Virage Coupe 2012", "Audi RS 4 Convertible 2008", "Audi A5 Coupe 2012",
    "Audi TTS Coupe 2012", "Audi R8 Coupe 2012", "Audi V8 Sedan 1994", "Audi 100 Sedan 1994",
    "Audi 100 Wagon 1994", "Audi TT Hatchback 2011", "Audi S6 Sedan 2011",
    "Audi S5 Convertible 2012", "Audi S5 Coupe 2012", "Audi S4 Sedan 2012",
    "Audi S4 Sedan 2007", "Audi TT RS Coupe 2012", "BMW ActiveHybrid 5 Sedan 2012",
    "BMW 1 Series Convertible 2012", "BMW 1 Series Coupe 2012", "BMW 3 Series Sedan 2012",
    "BMW 3 Series Wagon 2012", "BMW 6 Series Convertible 2007", "BMW X5 SUV 2007",
    "BMW X6 SUV 2012", "BMW M3 Coupe 2012", "BMW M5 Sedan 2010", "BMW M6 Convertible 2010",
    "BMW X3 SUV 2012", "BMW Z4 Convertible 2012", "Bentley Continental Supersports Conv. Convertible 2012",
    "Bentley Arnage Sedan 2009", "Bentley Mulsanne Sedan 2011", "Bentley Continental GT Coupe 2012",
    "Bentley Continental GT Coupe 2007", "Bentley Continental Flying Spur Sedan 2007",
    "Bugatti Veyron 16.4 Convertible 2009", "Bugatti Veyron 16.4 Coupe 2009",
    "Buick Regal GS 2012", "Buick Rainier SUV 2007", "Buick Verano Sedan 2012",
    "Buick Enclave SUV 2012", "Cadillac CTS-V Sedan 2012", "Cadillac SRX SUV 2012",
    "Cadillac Escalade EXT Crew Cab 2007", "Chevrolet Silverado 1500 Hybrid Crew Cab 2012",
    "Chevrolet Corvette Convertible 2012", "Chevrolet Corvette ZR1 2012",
    "Chevrolet Corvette Ron Fellows Edition Z06 2007", "Chevrolet Traverse SUV 2012",
    "Chevrolet Camaro Convertible 2012", "Chevrolet HHR SS 2010", "Chevrolet Impala Sedan 2007",
    "Chevrolet Tahoe Hybrid SUV 2012", "Chevrolet Sonic Sedan 2012", "Chevrolet Express Cargo Van 2007",
    "Chevrolet Avalanche Crew Cab 2012", "Chevrolet Cobalt SS 2010", "Chevrolet Malibu Hybrid Sedan 2010",
    "Chevrolet TrailBlazer SS 2009", "Chevrolet Silverado 2500HD Regular Cab 2012",
    "Chevrolet Silverado 1500 Classic Extended Cab 2007", "Chevrolet Express Van 2007",
    "Chevrolet Monte Carlo Coupe 2007", "Chevrolet Malibu Sedan 2007",
    "Chevrolet Silverado 1500 Extended Cab 2012", "Chevrolet Silverado 1500 Regular Cab 2012",
    "Chrysler Aspen SUV 2009", "Chrysler Sebring Convertible 2010",
    "Chrysler Town and Country Minivan 2012", "Chrysler 300 SRT-8 2010",
    "Chrysler Crossfire Convertible 2008", "Chrysler PT Cruiser Convertible 2008",
    "Daewoo Nubira Wagon 2002", "Dodge Caliber Wagon 2012", "Dodge Caliber Wagon 2007",
    "Dodge Caravan Minivan 1997", "Dodge Ram Pickup 3500 Crew Cab 2010",
    "Dodge Ram Pickup 3500 Quad Cab 2009", "Dodge Sprinter Cargo Van 2009",
    "Dodge Journey SUV 2012", "Dodge Dakota Crew Cab 2010", "Dodge Dakota Club Cab 2007",
    "Dodge Magnum Wagon 2008", "Dodge Challenger SRT8 2011", "Dodge Durango SUV 2012",
    "Dodge Durango SUV 2007", "Dodge Charger Sedan 2012", "Dodge Charger SRT-8 2009",
    "Eagle Talon Hatchback 1998", "FIAT 500 Abarth 2012", "FIAT 500 Convertible 2012",
    "Ferrari FF Coupe 2012", "Ferrari California Convertible 2012",
    "Ferrari 458 Italia Convertible 2012", "Ferrari 458 Italia Coupe 2012",
    "Fisker Karma Sedan 2012", "Ford F-450 Super Duty Crew Cab 2012",
    "Ford Mustang Convertible 2007", "Ford Freestar Minivan 2007", "Ford Expedition EL SUV 2009",
    "Ford Edge SUV 2012", "Ford Ranger SuperCab 2011", "Ford GT Coupe 2006",
    "Ford F-150 Regular Cab 2012", "Ford F-150 Regular Cab 2007", "Ford Focus Sedan 2007",
    "Ford E-Series Wagon Van 2012", "Ford Fiesta Sedan 2012", "GMC Terrain SUV 2012",
    "GMC Savana Van 2012", "GMC Yukon Hybrid SUV 2012", "GMC Acadia SUV 2012",
    "GMC Canyon Extended Cab 2012", "Geo Metro Convertible 1993", "HUMMER H3T Crew Cab 2010",
    "HUMMER H2 SUT Crew Cab 2009", "Honda Odyssey Minivan 2012", "Honda Odyssey Minivan 2007",
    "Honda Accord Coupe 2012", "Honda Accord Sedan 2012", "Hyundai Veloster Hatchback 2012",
    "Hyundai Santa Fe SUV 2012", "Hyundai Tucson SUV 2012", "Hyundai Veracruz SUV 2012",
    "Hyundai Sonata Hybrid Sedan 2012", "Hyundai Elantra Sedan 2007", "Hyundai Accent Sedan 2012",
    "Hyundai Genesis Sedan 2012", "Hyundai Sonata Sedan 2012", "Hyundai Elantra Touring Hatchback 2012",
    "Hyundai Azera Sedan 2012", "Infiniti G Coupe IPL 2012", "Infiniti QX56 SUV 2011",
    "Isuzu Ascender SUV 2008", "Jaguar XK XKR 2012", "Jeep Patriot SUV 2012",
    "Jeep Wrangler SUV 2012", "Jeep Liberty SUV 2012", "Jeep Grand Cherokee SUV 2012",
    "Jeep Compass SUV 2012", "Lamborghini Reventon Coupe 2008", "Lamborghini Aventador Coupe 2012",
    "Lamborghini Gallardo LP 570-4 Superleggera 2012", "Lamborghini Diablo Coupe 2001",
    "Land Rover Range Rover SUV 2012", "Land Rover LR2 SUV 2012", "Lincoln Town Car Sedan 2011",
    "MINI Cooper Roadster Convertible 2012", "Maybach Landaulet Convertible 2012",
    "Mazda Tribute SUV 2011", "McLaren MP4-12C Coupe 2012", "Mercedes-Benz 300-Class Convertible 1993",
    "Mercedes-Benz C-Class Sedan 2012", "Mercedes-Benz SL-Class Coupe 2009",
    "Mercedes-Benz E-Class Sedan 2012", "Mercedes-Benz S-Class Sedan 2012",
    "Mercedes-Benz Sprinter Van 2012", "Mitsubishi Lancer Sedan 2012", "Nissan Leaf Hatchback 2012",
    "Nissan NV Passenger Van 2012", "Nissan Juke Hatchback 2012", "Nissan 240SX Coupe 1998",
    "Plymouth Neon Coupe 1999", "Porsche Panamera Sedan 2012", "Ram C/V Cargo Van Minivan 2012",
    "Rolls-Royce Phantom Drophead Coupe Convertible 2012", "Rolls-Royce Ghost Sedan 2012",
    "Rolls-Royce Phantom Sedan 2012", "Scion xD Hatchback 2012", "Spyker C8 Convertible 2009",
    "Spyker C8 Coupe 2009", "Suzuki Aerio Sedan 2007", "Suzuki Kizashi Sedan 2012",
    "Suzuki SX4 Hatchback 2012", "Suzuki SX4 Sedan 2012", "Tesla Model S Sedan 2012",
    "Toyota Sequoia SUV 2012", "Toyota Camry Sedan 2012", "Toyota Corolla Sedan 2012",
    "Toyota 4Runner SUV 2012", "Volkswagen Golf Hatchback 2012", "Volkswagen Golf Hatchback 1991",
    "Volkswagen Beetle Hatchback 2012", "Volvo C30 Hatchback 2012", "Volvo 240 Sedan 1993",
    "Volvo XC90 SUV 2007", "smart fortwo Convertible 2012"
]

FLOWERS102_CLASSNAMES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea",
    "english marigold", "tiger lily", "moon orchid", "bird of paradise", "monkshood",
    "globe thistle", "snapdragon", "colt's foot", "king protea", "spear thistle",
    "yellow iris", "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
    "giant white arum lily", "fire lily", "pincushion flower", "fritillary", "red ginger",
    "grape hyacinth", "corn poppy", "prince of wales feathers", "stemless gentian",
    "artichoke", "sweet william", "carnation", "garden phlox", "love in the mist",
    "mexican aster", "alpine sea holly", "ruby-lipped cattleya", "cape flower",
    "great masterwort", "siam tulip", "lenten rose", "barbeton daisy", "daffodil",
    "sword lily", "poinsettia", "bolero deep blue", "wallflower", "marigold",
    "buttercup", "oxeye daisy", "common dandelion", "petunia", "wild pansy", "primula",
    "sunflower", "pelargonium", "bishop of llandaff", "gaura", "geranium",
    "orange dahlia", "pink-yellow dahlia?", "cautleya spicata", "japanese anemone",
    "black-eyed susan", "silverbush", "californian poppy", "osteospermum",
    "spring crocus", "bearded iris", "windflower", "tree poppy", "gazania", "azalea",
    "water lily", "rose", "thorn apple", "morning glory", "passion flower", "lotus",
    "toad lily", "anthurium", "frangipani", "clematis", "hibiscus", "columbine",
    "desert-rose", "tree mallow", "magnolia", "cyclamen", "watercress", "canna lily",
    "hippeastrum", "bee balm", "ball moss", "foxglove", "bougainvillea", "camellia",
    "mallow", "mexican petunia", "bromelia", "blanket flower", "trumpet creeper",
    "blackberry lily"
]

FOOD101_CLASSNAMES = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
    "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
    "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
    "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
    "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
    "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
    "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
    "mussels", "nachos", "omelette", "onion_rings", "oysters",
    "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
    "pho", "pizza", "pork_chop", "poutine", "prime_rib",
    "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto",
    "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
    "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
    "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare",
    "waffles"
]

RESISC45_CLASSNAMES = [
    'airplane', 'airport', 'baseball_diamond', 'basketball_court', 'beach',
    'bridge', 'chaparral', 'church', 'circular_farmland', 'cloud',
    'commercial_area', 'dense_residential', 'desert', 'forest', 'freeway',
    'golf_course', 'ground_track_field', 'harbor', 'industrial_area',
    'intersection', 'island', 'lake', 'meadow', 'medium_residential',
    'mobile_home_park', 'mountain', 'overpass', 'palace', 'parking_lot',
    'railway', 'railway_station', 'rectangular_farmland', 'river',
    'roundabout', 'runway', 'sea_ice', 'ship', 'snowberg',
    'sparse_residential', 'stadium', 'storage_tank', 'tennis_court',
    'terrace', 'thermal_power_station', 'wetland'
]


TASK_TO_CLASSNAMES: Dict[str, List[str]] = {
    "mnist": MNIST_CLASSNAMES,
    "kmnist": KMNIST_CLASSNAMES,
    "svhn": SVHN_CLASSNAMES,
    "emnist-mnist": EMNIST_MNIST_CLASSNAMES,
    "emnist-letters": EMNIST_LETTERS_CLASSNAMES,
    "fashion-mnist": FASHION_MNIST_CLASSNAMES,
    "cifar10": CIFAR10_CLASSNAMES,
    "cifar100": CIFAR100_CLASSNAMES,
    "stl10": STL10_CLASSNAMES,
    "dtd": DTD_CLASSNAMES,
    "eurosat": EUROSAT_CLASSNAMES,
    "gtsrb": GTSRB_CLASSNAMES,
    "fer2013": FER2013_CLASSNAMES,
    "rendered-sst2": RENDERED_SST2_CLASSNAMES,
    "pcam": PCAM_CLASSNAMES,
    "resisc45": RESISC45_CLASSNAMES,
    "stanford-cars": STANFORD_CARS_CLASSNAMES,
    "flowers102": FLOWERS102_CLASSNAMES,
    "food101": FOOD101_CLASSNAMES,
    "sun397": SUN397_CLASSNAMES,
}


def get_classnames(task_name: str) -> List[str]:
    if task_name not in TASK_TO_CLASSNAMES:
        raise ValueError(f"Unknown task '{task_name}'. Available: {sorted(TASK_TO_CLASSNAMES)}")
    classnames = TASK_TO_CLASSNAMES[task_name]
    if not classnames:
        raise ValueError(f"Class names for '{task_name}' are empty.")
    return classnames


def get_num_classes(task_name: str) -> int:
    return len(get_classnames(task_name))
