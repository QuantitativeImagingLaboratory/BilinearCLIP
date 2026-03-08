import random
import torch
import numpy as np
import os
import yaml
from tqdm import tqdm
import clip

def seed_everything(seed=42):
    print("seeding Everyting!")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

def get_optimizer(name):
    print(f"Optimizer: {name}")
    if name.lower() == "adam":
        return torch.optim.Adam
    elif name.lower() == "adamw":
        return torch.optim.AdamW
    elif name.lower() == "sgd":
        return torch.optim.SGD

def get_optimizer_params(training_params, model):
    lr = float(training_params["lr"])
    weight_decay = float(training_params["weight_decay"])
    print("W", "lr:", lr, "weight_decay:", weight_decay)
    params = [{'params': model.W, 'lr': lr, 'weight_decay': weight_decay}]

    print(f"---------Trainable Params---------------: {len(params)}")
    return params


def get_scheduler(training_prams, optimizer, epochs):
    if training_prams["lr_scheduler"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif training_prams["lr_scheduler"] == "cosine+warmup":
        print("Custom Scheduler: Linear Warmup (5epochs) + Cosine Annealing")
        import math
        def get_lr_lambda(epoch):
            warmup_epochs = 10
            total_epochs = epochs
            assert warmup_epochs <= total_epochs, "Scheduler Error"
            # if warmup_epochs < total_epochs:
            #     print(f"Scheduler error: Warmup epochs: {warmup_epochs}, total epochs: {total_epochs}")
            # 1. Linear Warmup Phase
            if epoch < warmup_epochs:
                return float(epoch + 1) / warmup_epochs
            # 2. Cosine Decay Phase
            else:
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_lambda)
        return scheduler
    elif training_prams["lr_scheduler"] == "steplr":
        print(f"Using StepLR Scheduler")
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return scheduler

def get_flower_names(i):
    flower_names = [
        "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea", "english marigold",
        "tiger lily", "moon orchid", "bird of paradise", "monkshood", "globe thistle", "snapdragon",
        "colt's foot", "king protea", "spear thistle", "yellow iris", "globe-flower", "purple coneflower",
        "peruvian lily", "balloon flower", "giant white arum lily", "fire lily", "pincushion flower",
        "fritillary", "red ginger", "grape hyacinth", "corn poppy", "prince of wales feathers",
        "stemless gentian", "artichoke", "sweet william", "carnation", "garden phlox", "love in the mist",
        "mexican aster", "alpine sea holly", "ruby-lipped cattleya", "cape flower", "great masterwort",
        "siam tulip", "lenten rose", "barbeton daisy", "daffodil", "sword lily", "poinsettia",
        "bolero deep blue", "wallflower", "marigold", "buttercup", "oxeye daisy", "common dandelion",
        "mexican petunia", "wild pansy", "primula", "sunflower", "pelargonium", "bishop of llandaff",
        "gaura", "geranium", "orange dahlia", "pink-yellow dahlia", "cautleya spicata", "japanese anemone",
        "black-eyed susan", "silverbush", "californian poppy", "osteospermum", "spring crocus",
        "bearded iris", "windflower", "tree poppy", "gazania", "azalea", "water lily", "rose",
        "thorn apple", "morning glory", "passion flower", "lotus", "toad lily", "anthurium",
        "frangipani", "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow", "magnolia",
        "cyclamen", "watercress", "canna lily", "hippeastrum", "bee balm", "ball moss", "foxglove",
        "bougainvillea", "camellia", "mallow", "mexican petunia", "bromelia", "blanket flower",
        "trumpet creeper", "blackberry lily"
    ]

    return flower_names[i]

def get_eurosat_classes(input_list):
    class_map = {
        "annualcrop": "annual crop land",
        "forest": "forest",
        "herbaceousvegetation": "herbaceous vegetation land",
        "highway": "highway or road",
        "industrial": "industrial buildings",
        "pasture": "pasture land",
        "permanentcrop": "permanent crop land",
        "residential": "residential buildings",
        "river": "river",
        "sealake": "sea or lake"
    }

    return [class_map[k.lower()] for k in input_list]

def get_imagenet_templates():
    # imagenet_templates = ["a bad photo of a {}.", "a photo of many {}.", "a sculpture of a {}.", "a photo of the hard to see {}.", "a low resolution photo of the {}.", "a rendering of a {}.", "graffiti of a {}.", "a bad photo of the {}.", "a cropped photo of the {}.", "a tattoo of a {}.", "the embroidered {}.", "a photo of a hard to see {}.", "a bright photo of a {}.", "a photo of a clean {}.", "a photo of a dirty {}.", "a dark photo of the {}.", "a drawing of a {}.", "a photo of my {}.", "the plastic {}.", "a photo of the cool {}.", "a close-up photo of a {}.", "a black and white photo of the {}.", "a painting of the {}.", "a painting of a {}.", "a pixelated photo of the {}.", "a sculpture of the {}.", "a bright photo of the {}.", "a cropped photo of a {}.", "a plastic {}.", "a photo of the dirty {}.", "a jpeg corrupted photo of a {}.", "a blurry photo of the {}.", "a photo of the {}.", "a good photo of the {}.", "a rendering of the {}.", "a {} in a video game.", "a photo of one {}.", "a doodle of a {}.", "a close-up photo of the {}.", "a photo of a {}.", "the origami {}.", "the {} in a video game.", "a sketch of a {}.", "a doodle of the {}.", "a origami {}.", "a low resolution photo of a {}.", "the toy {}.", "a rendition of the {}.", "a photo of the clean {}.", "a photo of a large {}.", "a rendition of a {}.", "a photo of a nice {}.", "a photo of a weird {}.", "a blurry photo of a {}.", "a cartoon {}.", "art of a {}.", "a sketch of the {}.", "a embroidered {}.", "a pixelated photo of a {}.", "itap of the {}.", "a jpeg corrupted photo of the {}.", "a good photo of a {}.", "a plushie {}.", "a photo of the nice {}.", "a photo of the small {}.", "a photo of the weird {}.", "the cartoon {}.", "art of the {}.", "a drawing of the {}.", "a photo of the large {}.", "a black and white photo of a {}.", "the plushie {}.", "a dark photo of a {}.", "itap of a {}.", "graffiti of the {}.", "a toy {}.", "itap of my {}.", "a photo of a cool {}.", "a photo of a small {}.", "a tattoo of the {}."]
    imagenet_templates = ["itap of a {}.",
                          "a bad photo of the {}.",
                          "a origami {}.",
                          "a photo of the large {}.",
                          "a {} in a video game.",
                          "art of the {}.",
                          "a photo of the small {}."]
    return imagenet_templates

def get_imagenet_classes():
    imagenet_classes = ["tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray",
                        "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco",
                        "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper",
                        "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander",
                        "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog",
                        "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin",
                        "box turtle", "banded gecko", "green iguana", "Carolina anole",
                        "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard",
                        "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile",
                        "American alligator", "triceratops", "worm snake", "ring-necked snake",
                        "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake",
                        "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra",
                        "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake",
                        "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider",
                        "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider",
                        "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl",
                        "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet",
                        "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck",
                        "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby",
                        "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch",
                        "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab",
                        "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab",
                        "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron",
                        "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot",
                        "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher",
                        "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion",
                        "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel",
                        "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle",
                        "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound",
                        "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound",
                        "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound",
                        "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier",
                        "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier",
                        "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier",
                        "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier",
                        "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer",
                        "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier",
                        "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier",
                        "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever",
                        "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla",
                        "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel",
                        "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel",
                        "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard",
                        "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie",
                        "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann",
                        "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog",
                        "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff",
                        "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky",
                        "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog",
                        "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon",
                        "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle",
                        "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf",
                        "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox",
                        "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat",
                        "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger",
                        "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose",
                        "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle",
                        "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper",
                        "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper",
                        "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly",
                        "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly",
                        "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit",
                        "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse",
                        "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison",
                        "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)",
                        "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat",
                        "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan",
                        "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque",
                        "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin",
                        "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey",
                        "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda",
                        "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish",
                        "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown",
                        "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance",
                        "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle",
                        "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo",
                        "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel",
                        "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel",
                        "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)",
                        "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini",
                        "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet",
                        "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra",
                        "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest",
                        "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe",
                        "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton",
                        "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran",
                        "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw",
                        "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking",
                        "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker",
                        "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard",
                        "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot",
                        "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed",
                        "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer",
                        "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table",
                        "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig",
                        "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar",
                        "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder",
                        "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute",
                        "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed",
                        "freight car", "French horn", "frying pan", "fur coat", "garbage truck",
                        "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola",
                        "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine",
                        "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer",
                        "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet",
                        "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar",
                        "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep",
                        "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat",
                        "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library",
                        "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion",
                        "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag",
                        "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask",
                        "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone",
                        "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile",
                        "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor",
                        "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa",
                        "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail",
                        "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina",
                        "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart",
                        "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush",
                        "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench",
                        "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case",
                        "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube",
                        "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball",
                        "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag",
                        "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho",
                        "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug",
                        "printer", "prison", "missile", "projector", "hockey puck", "punching bag", "purse", "quill",
                        "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel",
                        "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator",
                        "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser",
                        "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal",
                        "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard",
                        "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store",
                        "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap",
                        "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door",
                        "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock",
                        "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater",
                        "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight",
                        "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf",
                        "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa",
                        "submarine", "suit", "sundial", "sunglasses", "sunglasses", "sunscreen", "suspension bridge",
                        "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe",
                        "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball",
                        "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof",
                        "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store",
                        "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod",
                        "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard",
                        "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling",
                        "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball",
                        "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink",
                        "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle",
                        "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing",
                        "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website",
                        "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu",
                        "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette",
                        "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli",
                        "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber",
                        "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange",
                        "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate",
                        "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito",
                        "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef",
                        "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player",
                        "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn",
                        "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom",
                        "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"]

    return imagenet_classes


def get_zeroshot_weights(model, classnames, device, templates=None):
    if templates is None:
        templates = get_imagenet_templates()

    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):

            texts = [template.format(classname) for template in templates]
            texts = clip.tokenize(texts).to(device)

            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

            class_embedding = class_embeddings.mean(dim=0)

            class_embedding /= class_embedding.norm()

            zeroshot_weights.append(class_embedding)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights

def get_zeroshot_weights_for_sun397(model, classes, prompt, device):
    # 1. Create all templates
    templates = [prompt % c for c in classes]

    # 2. Encode in chunks to prevent OOM
    all_text_features = []
    chunk_size = 128  # Adjust based on your GPU (128-512 is usually safe)

    for i in range(0, len(templates), chunk_size):
        chunk = templates[i: i + chunk_size]
        text_tokens = clip.tokenize(chunk).to(device)

        with torch.no_grad():
            chunk_features = model.encode_text(text_tokens)
            chunk_features /= chunk_features.norm(dim=-1, keepdim=True)
            all_text_features.append(chunk_features)

    # 3. Aggregate (average) features for ensembling
    # Reshape back to [num_classes, num_templates, embedding_dim]
    text_features = torch.cat(all_text_features, dim=0)
    text_features = text_features.view(len(classes), -1, text_features.shape[-1])
    text_features = text_features.mean(dim=1)  # Mean over templates
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features


def get_config_file(dataset, num_shot, backbone, vlm="clip", ablation=None):

    assert backbone.lower() in ["rn50", "vit16", "vit32"], f"backbone {backbone} not supported"

    CONFIG_FOLDER = "configs"
    if vlm == "siglip":
        CONFIG_FOLDER_FULL = os.path.join(CONFIG_FOLDER, "siglip")
    elif vlm == "clip":
        CONFIG_FOLDER_FULL = os.path.join(CONFIG_FOLDER, "clip")

    config_file = os.path.join(CONFIG_FOLDER_FULL, f"{dataset}.yml")
    print(f"Loading config from {config_file}")
    cfg = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
    # if dataset == "imagenet":
    #     config_file = os.path.join(CONFIG_FOLDER, "imagenet.yml")
    #     cfg = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)

    if ablation is None:
        config_name = f"{num_shot}_shot_{backbone.lower()}"
    else:
        config_name = f"{num_shot}_shot_{backbone.lower()}_ablation_{ablation}"
    print(f"Loading config from {config_name}.")
    return cfg[config_name]
