
from __future__ import print_function
import torch
import numpy as np
import os
from models.vgg import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from models.Linear5 import Linear5
from models.LeNet5 import LeNet5
from models.inceptionresnetv2 import inceptionresnetv2
from models.vit import vit_b_16, vit_b_32,  vit_l_16, vit_l_32, vit_h_14
from torchvision import datasets
from torchvision.transforms import v2
import torch.nn.functional as F
import torch.distributions.categorical as cat
import torch.distributions.dirichlet as diri
from torch.utils.data import Dataset
import random
np.set_printoptions(suppress=True)

    
def get_dataset(args):
    if "fmnist" in args.dataset.lower():
        train_transform=v2.Compose([
            v2.ToTensor(),
            v2.Normalize((0.2860,), (0.3530,)),
            ])

        test_transform=v2.Compose([
            v2.ToTensor(),
            v2.Normalize((0.2860,), (0.3530,))
            ])
        if args.dataset.lower() =="fmnist":
            if args.train_transform:
                dataset1 = datasets.FashionMNIST(args.data_path, train=True, download=True, transform=train_transform)
            else:
                dataset1 = datasets.FashionMNIST(args.data_path, train=True, download=True, transform=test_transform)
            dataset2 = datasets.FashionMNIST(args.data_path, train=False, transform=test_transform)
            args.num_classes = 10
            args.class_label_names = [i for i in range(10)]
        else:
            raise ValueError 
    elif "mnist" in args.dataset.lower():
        train_transform=v2.Compose([
            v2.ToTensor(),
            v2.Normalize((0.1307,), (0.3081,)),
            ])
        
        test_transform=v2.Compose([
            v2.ToTensor(),
            v2.Normalize((0.1307,), (0.3081,))
            ])
        if args.dataset.lower() =="mnist":
            if args.train_transform:
                dataset1 = datasets.MNIST(args.data_path, train=True, download=True, transform=train_transform)
            else:
                dataset1 = datasets.MNIST(args.data_path, train=True, download=True, transform=test_transform)
            dataset2 = datasets.MNIST(args.data_path, train=False, transform=test_transform)
            args.num_classes = 10
            args.class_label_names = [i for i in range(10)]
        else:
            raise ValueError 
    elif "cifar" in args.dataset.lower():
        train_transform=v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.RandomCrop(32, 4),
            v2.ToTensor(),
            v2.Normalize(mean=[0.491, 0.482, 0.447],
                                        std=[0.247, 0.243, 0.262]),      
            ])
        
        test_transform=v2.Compose([
            v2.ToTensor(),
            v2.Normalize(mean=[0.491, 0.482, 0.447],
                                        std=[0.247, 0.243, 0.262]),   
            ])
        if args.dataset.lower() == "cifar10":
            if args.train_transform:
                dataset1 = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=train_transform)
            else:
                dataset1 = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=test_transform)
            dataset2 = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=test_transform)
            args.num_classes=10
            args.class_label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        elif args.dataset.lower() == "cifar100":
            if args.train_transform:
                dataset1 = datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=train_transform)
            else:
                dataset1 = datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=test_transform)
            dataset2 = datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=test_transform)
            args.num_classes=100
            args.class_label_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'computer_keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
        else:
            raise ValueError
        
    elif "imagenet" in args.dataset.lower() or "webvision" in args.dataset.lower():
        if args.arch.lower() == "inceptionresnetv2":
            args.resize_image = 299
        else:
            args.resize_image = 224
        train_transform=v2.Compose([ 
            v2.RandomResizedCrop((args.resize_image,args.resize_image)),
            v2.RandomHorizontalFlip(),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]),    
            ])

        test_transform=v2.Compose([
            v2.Resize((args.resize_image,args.resize_image)),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])     
            ])
        if args.dataset.lower() == "imagenette":
            if args.train_transform:
                dataset1 = datasets.ImageFolder(root=os.path.join(args.data_path, "imagenette","train"), transform=train_transform)
            else:
                dataset1 = datasets.ImageFolder(root=os.path.join(args.data_path, "imagenette","train"), transform=test_transform)
            dataset2 = datasets.ImageFolder(root=os.path.join(args.data_path, "imagenette","val"), transform=test_transform)
            args.num_classes = 10
            args.class_label_names = ["bench", "English springer", "cassette player", "chain saw", "church", "French horn", "garbage truck", "gas pump", "golf ball", "parachute"]
        else:
            args.num_classes = 1000
            args.class_label_names = ['tench, Tinca tinca', 'goldfish, Carassius auratus', 'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias', 'tiger shark, Galeocerdo cuvieri', 'hammerhead, hammerhead shark', 'electric ray, crampfish, numbfish, torpedo', 'stingray', 'cock', 'hen', 'ostrich, Struthio camelus', 'brambling, Fringilla montifringilla', 'goldfinch, Carduelis carduelis', 'house finch, linnet, Carpodacus mexicanus', 'junco, snowbird', 'indigo bunting, indigo finch, indigo bird, Passerina cyanea', 'robin, American robin, Turdus migratorius', 'bulbul', 'jay', 'magpie', 'chickadee', 'water ouzel, dipper', 'kite', 'bald eagle, American eagle, Haliaeetus leucocephalus', 'vulture', 'great grey owl, great gray owl, Strix nebulosa', 'European fire salamander, Salamandra salamandra', 'common newt, Triturus vulgaris', 'eft', 'spotted salamander, Ambystoma maculatum', 'axolotl, mud puppy, Ambystoma mexicanum', 'bullfrog, Rana catesbeiana', 'tree frog, tree-frog', 'tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui', 'loggerhead, loggerhead turtle, Caretta caretta', 'leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea', 'mud turtle', 'terrapin', 'box turtle, box tortoise', 'banded gecko', 'common iguana, iguana, Iguana iguana', 'American chameleon, anole, Anolis carolinensis', 'whiptail, whiptail lizard', 'agama', 'frilled lizard, Chlamydosaurus kingi', 'alligator lizard', 'Gila monster, Heloderma suspectum', 'green lizard, Lacerta viridis', 'African chameleon, Chamaeleo chamaeleon', 'Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis', 'African crocodile, Nile crocodile, Crocodylus niloticus', 'American alligator, Alligator mississipiensis', 'triceratops', 'thunder snake, worm snake, Carphophis amoenus', 'ringneck snake, ring-necked snake, ring snake', 'hognose snake, puff adder, sand viper', 'green snake, grass snake', 'king snake, kingsnake', 'garter snake, grass snake', 'water snake', 'vine snake', 'night snake, Hypsiglena torquata', 'boa constrictor, Constrictor constrictor', 'rock python, rock snake, Python sebae', 'Indian cobra, Naja naja', 'green mamba', 'sea snake', 'horned viper, cerastes, sand viper, horned asp, Cerastes cornutus', 'diamondback, diamondback rattlesnake, Crotalus adamanteus', 'sidewinder, horned rattlesnake, Crotalus cerastes', 'trilobite', 'harvestman, daddy longlegs, Phalangium opilio', 'scorpion', 'black and gold garden spider, Argiope aurantia', 'barn spider, Araneus cavaticus', 'garden spider, Aranea diademata', 'black widow, Latrodectus mactans', 'tarantula', 'wolf spider, hunting spider', 'tick', 'centipede', 'black grouse', 'ptarmigan', 'ruffed grouse, partridge, Bonasa umbellus', 'prairie chicken, prairie grouse, prairie fowl', 'peacock', 'quail', 'partridge', 'African grey, African gray, Psittacus erithacus', 'macaw', 'sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita', 'lorikeet', 'coucal', 'bee eater', 'hornbill', 'hummingbird', 'jacamar', 'toucan', 'drake', 'red-breasted merganser, Mergus serrator', 'goose', 'black swan, Cygnus atratus', 'tusker', 'echidna, spiny anteater, anteater', 'platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus', 'wallaby, brush kangaroo', 'koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus', 'wombat', 'jellyfish', 'sea anemone, anemone', 'brain coral', 'flatworm, platyhelminth', 'nematode, nematode worm, roundworm', 'conch', 'snail', 'slug', 'sea slug, nudibranch', 'chiton, coat-of-mail shell, sea cradle, polyplacophore', 'chambered nautilus, pearly nautilus, nautilus', 'Dungeness crab, Cancer magister', 'rock crab, Cancer irroratus', 'fiddler crab', 'king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica', 'American lobster, Northern lobster, Maine lobster, Homarus americanus', 'spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish', 'crayfish, crawfish, crawdad, crawdaddy', 'hermit crab', 'isopod', 'white stork, Ciconia ciconia', 'black stork, Ciconia nigra', 'spoonbill', 'flamingo', 'little blue heron, Egretta caerulea', 'American egret, great white heron, Egretta albus', 'bittern', 'crane', 'limpkin, Aramus pictus', 'European gallinule, Porphyrio porphyrio', 'American coot, marsh hen, mud hen, water hen, Fulica americana', 'bustard', 'ruddy turnstone, Arenaria interpres', 'red-backed sandpiper, dunlin, Erolia alpina', 'redshank, Tringa totanus', 'dowitcher', 'oystercatcher, oyster catcher', 'pelican', 'king penguin, Aptenodytes patagonica', 'albatross, mollymawk', 'grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus', 'killer whale, killer, orca, grampus, sea wolf, Orcinus orca', 'dugong, Dugong dugon', 'sea lion', 'Chihuahua', 'Japanese spaniel', 'Maltese dog, Maltese terrier, Maltese', 'Pekinese, Pekingese, Peke', 'Shih-Tzu', 'Blenheim spaniel', 'papillon', 'toy terrier', 'Rhodesian ridgeback', 'Afghan hound, Afghan', 'basset, basset hound', 'beagle', 'bloodhound, sleuthhound', 'bluetick', 'black-and-tan coonhound', 'Walker hound, Walker foxhound', 'English foxhound', 'redbone', 'borzoi, Russian wolfhound', 'Irish wolfhound', 'Italian greyhound', 'whippet', 'Ibizan hound, Ibizan Podenco', 'Norwegian elkhound, elkhound', 'otterhound, otter hound', 'Saluki, gazelle hound', 'Scottish deerhound, deerhound', 'Weimaraner', 'Staffordshire bullterrier, Staffordshire bull terrier', 'American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier', 'Bedlington terrier', 'Border terrier', 'Kerry blue terrier', 'Irish terrier', 'Norfolk terrier', 'Norwich terrier', 'Yorkshire terrier', 'wire-haired fox terrier', 'Lakeland terrier', 'Sealyham terrier, Sealyham', 'Airedale, Airedale terrier', 'cairn, cairn terrier', 'Australian terrier', 'Dandie Dinmont, Dandie Dinmont terrier', 'Boston bull, Boston terrier', 'miniature schnauzer', 'giant schnauzer', 'standard schnauzer', 'Scotch terrier, Scottish terrier, Scottie', 'Tibetan terrier, chrysanthemum dog', 'silky terrier, Sydney silky', 'soft-coated wheaten terrier', 'West Highland white terrier', 'Lhasa, Lhasa apso', 'flat-coated retriever', 'curly-coated retriever', 'golden retriever', 'Labrador retriever', 'Chesapeake Bay retriever', 'German short-haired pointer', 'vizsla, Hungarian pointer', 'English setter', 'Irish setter, red setter', 'Gordon setter', 'Brittany spaniel', 'clumber, clumber spaniel', 'English springer, English springer spaniel', 'Welsh springer spaniel', 'cocker spaniel, English cocker spaniel, cocker', 'Sussex spaniel', 'Irish water spaniel', 'kuvasz', 'schipperke', 'groenendael', 'malinois', 'briard', 'kelpie', 'komondor', 'Old English sheepdog, bobtail', 'Shetland sheepdog, Shetland sheep dog, Shetland', 'collie', 'Border collie', 'Bouvier des Flandres, Bouviers des Flandres', 'Rottweiler', 'German shepherd, German shepherd dog, German police dog, alsatian', 'Doberman, Doberman pinscher', 'miniature pinscher', 'Greater Swiss Mountain dog', 'Bernese mountain dog', 'Appenzeller', 'EntleBucher', 'boxer', 'bull mastiff', 'Tibetan mastiff', 'French bulldog', 'Great Dane', 'Saint Bernard, St Bernard', 'Eskimo dog, husky', 'malamute, malemute, Alaskan malamute', 'Siberian husky', 'dalmatian, coach dog, carriage dog', 'affenpinscher, monkey pinscher, monkey dog', 'basenji', 'pug, pug-dog', 'Leonberg', 'Newfoundland, Newfoundland dog', 'Great Pyrenees', 'Samoyed, Samoyede', 'Pomeranian', 'chow, chow chow', 'keeshond', 'Brabancon griffon', 'Pembroke, Pembroke Welsh corgi', 'Cardigan, Cardigan Welsh corgi', 'toy poodle', 'miniature poodle', 'standard poodle', 'Mexican hairless', 'timber wolf, grey wolf, gray wolf, Canis lupus', 'white wolf, Arctic wolf, Canis lupus tundrarum', 'red wolf, maned wolf, Canis rufus, Canis niger', 'coyote, prairie wolf, brush wolf, Canis latrans', 'dingo, warrigal, warragal, Canis dingo', 'dhole, Cuon alpinus', 'African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus', 'hyena, hyaena', 'red fox, Vulpes vulpes', 'kit fox, Vulpes macrotis', 'Arctic fox, white fox, Alopex lagopus', 'grey fox, gray fox, Urocyon cinereoargenteus', 'tabby, tabby cat', 'tiger cat', 'Persian cat', 'Siamese cat, Siamese', 'Egyptian cat', 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor', 'lynx, catamount', 'leopard, Panthera pardus', 'snow leopard, ounce, Panthera uncia', 'jaguar, panther, Panthera onca, Felis onca', 'lion, king of beasts, Panthera leo', 'tiger, Panthera tigris', 'cheetah, chetah, Acinonyx jubatus', 'brown bear, bruin, Ursus arctos', 'American black bear, black bear, Ursus americanus, Euarctos americanus', 'ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus', 'sloth bear, Melursus ursinus, Ursus ursinus', 'mongoose', 'meerkat, mierkat', 'tiger beetle', 'ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle', 'ground beetle, carabid beetle', 'long-horned beetle, longicorn, longicorn beetle', 'leaf beetle, chrysomelid', 'dung beetle', 'rhinoceros beetle', 'weevil', 'fly', 'bee', 'ant, emmet, pismire', 'grasshopper, hopper', 'cricket', 'walking stick, walkingstick, stick insect', 'cockroach, roach', 'mantis, mantid', 'cicada, cicala', 'leafhopper', 'lacewing, lacewing fly', "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk", 'damselfly', 'admiral', 'ringlet, ringlet butterfly', 'monarch, monarch butterfly, milkweed butterfly, Danaus plexippus', 'cabbage butterfly', 'sulphur butterfly, sulfur butterfly', 'lycaenid, lycaenid butterfly', 'starfish, sea star', 'sea urchin', 'sea cucumber, holothurian', 'wood rabbit, cottontail, cottontail rabbit', 'hare', 'Angora, Angora rabbit', 'hamster', 'porcupine, hedgehog', 'fox squirrel, eastern fox squirrel, Sciurus niger', 'marmot', 'beaver', 'guinea pig, Cavia cobaya', 'sorrel', 'zebra', 'hog, pig, grunter, squealer, Sus scrofa', 'wild boar, boar, Sus scrofa', 'warthog', 'hippopotamus, hippo, river horse, Hippopotamus amphibius', 'ox', 'water buffalo, water ox, Asiatic buffalo, Bubalus bubalis', 'bison', 'ram, tup', 'bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis', 'ibex, Capra ibex', 'hartebeest', 'impala, Aepyceros melampus', 'gazelle', 'Arabian camel, dromedary, Camelus dromedarius', 'llama', 'weasel', 'mink', 'polecat, fitch, foulmart, foumart, Mustela putorius', 'black-footed ferret, ferret, Mustela nigripes', 'otter', 'skunk, polecat, wood pussy', 'badger', 'armadillo', 'three-toed sloth, ai, Bradypus tridactylus', 'orangutan, orang, orangutang, Pongo pygmaeus', 'gorilla, Gorilla gorilla', 'chimpanzee, chimp, Pan troglodytes', 'gibbon, Hylobates lar', 'siamang, Hylobates syndactylus, Symphalangus syndactylus', 'guenon, guenon monkey', 'patas, hussar monkey, Erythrocebus patas', 'baboon', 'macaque', 'langur', 'colobus, colobus monkey', 'proboscis monkey, Nasalis larvatus', 'marmoset', 'capuchin, ringtail, Cebus capucinus', 'howler monkey, howler', 'titi, titi monkey', 'spider monkey, Ateles geoffroyi', 'squirrel monkey, Saimiri sciureus', 'Madagascar cat, ring-tailed lemur, Lemur catta', 'indri, indris, Indri indri, Indri brevicaudatus', 'Indian elephant, Elephas maximus', 'African elephant, Loxodonta africana', 'lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens', 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca', 'barracouta, snoek', 'eel', 'coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch', 'rock beauty, Holocanthus tricolor', 'anemone fish', 'sturgeon', 'gar, garfish, garpike, billfish, Lepisosteus osseus', 'lionfish', 'puffer, pufferfish, blowfish, globefish', 'abacus', 'abaya', "academic gown, academic robe, judge's robe", 'accordion, piano accordion, squeeze box', 'acoustic guitar', 'aircraft carrier, carrier, flattop, attack aircraft carrier', 'airliner', 'airship, dirigible', 'altar', 'ambulance', 'amphibian, amphibious vehicle', 'analog clock', 'apiary, bee house', 'apron', 'ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin', 'assault rifle, assault gun', 'backpack, back pack, knapsack, packsack, rucksack, haversack', 'bakery, bakeshop, bakehouse', 'balance beam, beam', 'balloon', 'ballpoint, ballpoint pen, ballpen, Biro', 'Band Aid', 'banjo', 'bannister, banister, balustrade, balusters, handrail', 'barbell', 'barber chair', 'barbershop', 'barn', 'barometer', 'barrel, cask', 'barrow, garden cart, lawn cart, wheelbarrow', 'baseball', 'basketball', 'bassinet', 'bassoon', 'bathing cap, swimming cap', 'bath towel', 'bathtub, bathing tub, bath, tub', 'beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon', 'beacon, lighthouse, beacon light, pharos', 'beaker', 'bearskin, busby, shako', 'beer bottle', 'beer glass', 'bell cote, bell cot', 'bib', 'bicycle-built-for-two, tandem bicycle, tandem', 'bikini, two-piece', 'binder, ring-binder', 'binoculars, field glasses, opera glasses', 'birdhouse', 'boathouse', 'bobsled, bobsleigh, bob', 'bolo tie, bolo, bola tie, bola', 'bonnet, poke bonnet', 'bookcase', 'bookshop, bookstore, bookstall', 'bottlecap', 'bow', 'bow tie, bow-tie, bowtie', 'brass, memorial tablet, plaque', 'brassiere, bra, bandeau', 'breakwater, groin, groyne, mole, bulwark, seawall, jetty', 'breastplate, aegis, egis', 'broom', 'bucket, pail', 'buckle', 'bulletproof vest', 'bullet train, bullet', 'butcher shop, meat market', 'cab, hack, taxi, taxicab', 'caldron, cauldron', 'candle, taper, wax light', 'cannon', 'canoe', 'can opener, tin opener', 'cardigan', 'car mirror', 'carousel, carrousel, merry-go-round, roundabout, whirligig', "carpenter's kit, tool kit", 'carton', 'car wheel', 'cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM', 'cassette', 'cassette player', 'castle', 'catamaran', 'CD player', 'cello, violoncello', 'cellular telephone, cellular phone, cellphone, cell, mobile phone', 'chain', 'chainlink fence', 'chain mail, ring mail, mail, chain armor, chain armour, ring armor, ring armour', 'chain saw, chainsaw', 'chest', 'chiffonier, commode', 'chime, bell, gong', 'china cabinet, china closet', 'Christmas stocking', 'church, church building', 'cinema, movie theater, movie theatre, movie house, picture palace', 'cleaver, meat cleaver, chopper', 'cliff dwelling', 'cloak', 'clog, geta, patten, sabot', 'cocktail shaker', 'coffee mug', 'coffeepot', 'coil, spiral, volute, whorl, helix', 'combination lock', 'computer keyboard, keypad', 'confectionery, confectionary, candy store', 'container ship, containership, container vessel', 'convertible', 'corkscrew, bottle screw', 'cornet, horn, trumpet, trump', 'cowboy boot', 'cowboy hat, ten-gallon hat', 'cradle', 'crane', 'crash helmet', 'crate', 'crib, cot', 'Crock Pot', 'croquet ball', 'crutch', 'cuirass', 'dam, dike, dyke', 'desk', 'desktop computer', 'dial telephone, dial phone', 'diaper, nappy, napkin', 'digital clock', 'digital watch', 'dining table, board', 'dishrag, dishcloth', 'dishwasher, dish washer, dishwashing machine', 'disk brake, disc brake', 'dock, dockage, docking facility', 'dogsled, dog sled, dog sleigh', 'dome', 'doormat, welcome mat', 'drilling platform, offshore rig', 'drum, membranophone, tympan', 'drumstick', 'dumbbell', 'Dutch oven', 'electric fan, blower', 'electric guitar', 'electric locomotive', 'entertainment center', 'envelope', 'espresso maker', 'face powder', 'feather boa, boa', 'file, file cabinet, filing cabinet', 'fireboat', 'fire engine, fire truck', 'fire screen, fireguard', 'flagpole, flagstaff', 'flute, transverse flute', 'folding chair', 'football helmet', 'forklift', 'fountain', 'fountain pen', 'four-poster', 'freight car', 'French horn, horn', 'frying pan, frypan, skillet', 'fur coat', 'garbage truck, dustcart', 'gasmask, respirator, gas helmet', 'gas pump, gasoline pump, petrol pump, island dispenser', 'goblet', 'go-kart', 'golf ball', 'golfcart, golf cart', 'gondola', 'gong, tam-tam', 'gown', 'grand piano, grand', 'greenhouse, nursery, glasshouse', 'grille, radiator grille', 'grocery store, grocery, food market, market', 'guillotine', 'hair slide', 'hair spray', 'half track', 'hammer', 'hamper', 'hand blower, blow dryer, blow drier, hair dryer, hair drier', 'hand-held computer, hand-held microcomputer', 'handkerchief, hankie, hanky, hankey', 'hard disc, hard disk, fixed disk', 'harmonica, mouth organ, harp, mouth harp', 'harp', 'harvester, reaper', 'hatchet', 'holster', 'home theater, home theatre', 'honeycomb', 'hook, claw', 'hoopskirt, crinoline', 'horizontal bar, high bar', 'horse cart, horse-cart', 'hourglass', 'iPod', 'iron, smoothing iron', "jack-o'-lantern", 'jean, blue jean, denim', 'jeep, landrover', 'jersey, T-shirt, tee shirt', 'jigsaw puzzle', 'jinrikisha, ricksha, rickshaw', 'joystick', 'kimono', 'knee pad', 'knot', 'lab coat, laboratory coat', 'ladle', 'lampshade, lamp shade', 'laptop, laptop computer', 'lawn mower, mower', 'lens cap, lens cover', 'letter opener, paper knife, paperknife', 'library', 'lifeboat', 'lighter, light, igniter, ignitor', 'limousine, limo', 'liner, ocean liner', 'lipstick, lip rouge', 'Loafer', 'lotion', 'loudspeaker, speaker, speaker unit, loudspeaker system, speaker system', "loupe, jeweler's loupe", 'lumbermill, sawmill', 'magnetic compass', 'mailbag, postbag', 'mailbox, letter box', 'maillot', 'maillot, tank suit', 'manhole cover', 'maraca', 'marimba, xylophone', 'mask', 'matchstick', 'maypole', 'maze, labyrinth', 'measuring cup', 'medicine chest, medicine cabinet', 'megalith, megalithic structure', 'microphone, mike', 'microwave, microwave oven', 'military uniform', 'milk can', 'minibus', 'miniskirt, mini', 'minivan', 'missile', 'mitten', 'mixing bowl', 'mobile home, manufactured home', 'Model T', 'modem', 'monastery', 'monitor', 'moped', 'mortar', 'mortarboard', 'mosque', 'mosquito net', 'motor scooter, scooter', 'mountain bike, all-terrain bike, off-roader', 'mountain tent', 'mouse, computer mouse', 'mousetrap', 'moving van', 'muzzle', 'nail', 'neck brace', 'necklace', 'nipple', 'notebook, notebook computer', 'obelisk', 'oboe, hautboy, hautbois', 'ocarina, sweet potato', 'odometer, hodometer, mileometer, milometer', 'oil filter', 'organ, pipe organ', 'oscilloscope, scope, cathode-ray oscilloscope, CRO', 'overskirt', 'oxcart', 'oxygen mask', 'packet', 'paddle, boat paddle', 'paddlewheel, paddle wheel', 'padlock', 'paintbrush', "pajama, pyjama, pj's, jammies", 'palace', 'panpipe, pandean pipe, syrinx', 'paper towel', 'parachute, chute', 'parallel bars, bars', 'park bench', 'parking meter', 'passenger car, coach, carriage', 'patio, terrace', 'pay-phone, pay-station', 'pedestal, plinth, footstall', 'pencil box, pencil case', 'pencil sharpener', 'perfume, essence', 'Petri dish', 'photocopier', 'pick, plectrum, plectron', 'pickelhaube', 'picket fence, paling', 'pickup, pickup truck', 'pier', 'piggy bank, penny bank', 'pill bottle', 'pillow', 'ping-pong ball', 'pinwheel', 'pirate, pirate ship', 'pitcher, ewer', "plane, carpenter's plane, woodworking plane", 'planetarium', 'plastic bag', 'plate rack', 'plow, plough', "plunger, plumber's helper", 'Polaroid camera, Polaroid Land camera', 'pole', 'police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria', 'poncho', 'pool table, billiard table, snooker table', 'pop bottle, soda bottle', 'pot, flowerpot', "potter's wheel", 'power drill', 'prayer rug, prayer mat', 'printer', 'prison, prison house', 'projectile, missile', 'projector', 'puck, hockey puck', 'punching bag, punch bag, punching ball, punchball', 'purse', 'quill, quill pen', 'quilt, comforter, comfort, puff', 'racer, race car, racing car', 'racket, racquet', 'radiator', 'radio, wireless', 'radio telescope, radio reflector', 'rain barrel', 'recreational vehicle, RV, R.V.', 'reel', 'reflex camera', 'refrigerator, icebox', 'remote control, remote', 'restaurant, eating house, eating place, eatery', 'revolver, six-gun, six-shooter', 'rifle', 'rocking chair, rocker', 'rotisserie', 'rubber eraser, rubber, pencil eraser', 'rugby ball', 'rule, ruler', 'running shoe', 'safe', 'safety pin', 'saltshaker, salt shaker', 'sandal', 'sarong', 'sax, saxophone', 'scabbard', 'scale, weighing machine', 'school bus', 'schooner', 'scoreboard', 'screen, CRT screen', 'screw', 'screwdriver', 'seat belt, seatbelt', 'sewing machine', 'shield, buckler', 'shoe shop, shoe-shop, shoe store', 'shoji', 'shopping basket', 'shopping cart', 'shovel', 'shower cap', 'shower curtain', 'ski', 'ski mask', 'sleeping bag', 'slide rule, slipstick', 'sliding door', 'slot, one-armed bandit', 'snorkel', 'snowmobile', 'snowplow, snowplough', 'soap dispenser', 'soccer ball', 'sock', 'solar dish, solar collector, solar furnace', 'sombrero', 'soup bowl', 'space bar', 'space heater', 'space shuttle', 'spatula', 'speedboat', "spider web, spider's web", 'spindle', 'sports car, sport car', 'spotlight, spot', 'stage', 'steam locomotive', 'steel arch bridge', 'steel drum', 'stethoscope', 'stole', 'stone wall', 'stopwatch, stop watch', 'stove', 'strainer', 'streetcar, tram, tramcar, trolley, trolley car', 'stretcher', 'studio couch, day bed', 'stupa, tope', 'submarine, pigboat, sub, U-boat', 'suit, suit of clothes', 'sundial', 'sunglass', 'sunglasses, dark glasses, shades', 'sunscreen, sunblock, sun blocker', 'suspension bridge', 'swab, swob, mop', 'sweatshirt', 'swimming trunks, bathing trunks', 'swing', 'switch, electric switch, electrical switch', 'syringe', 'table lamp', 'tank, army tank, armored combat vehicle, armoured combat vehicle', 'tape player', 'teapot', 'teddy, teddy bear', 'television, television system', 'tennis ball', 'thatch, thatched roof', 'theater curtain, theatre curtain', 'thimble', 'thresher, thrasher, threshing machine', 'throne', 'tile roof', 'toaster', 'tobacco shop, tobacconist shop, tobacconist', 'toilet seat', 'torch', 'totem pole', 'tow truck, tow car, wrecker', 'toyshop', 'tractor', 'trailer truck, tractor trailer, trucking rig, rig, articulated lorry, semi', 'tray', 'trench coat', 'tricycle, trike, velocipede', 'trimaran', 'tripod', 'triumphal arch', 'trolleybus, trolley coach, trackless trolley', 'trombone', 'tub, vat', 'turnstile', 'typewriter keyboard', 'umbrella', 'unicycle, monocycle', 'upright, upright piano', 'vacuum, vacuum cleaner', 'vase', 'vault', 'velvet', 'vending machine', 'vestment', 'viaduct', 'violin, fiddle', 'volleyball', 'waffle iron', 'wall clock', 'wallet, billfold, notecase, pocketbook', 'wardrobe, closet, press', 'warplane, military plane', 'washbasin, handbasin, washbowl, lavabo, wash-hand basin', 'washer, automatic washer, washing machine', 'water bottle', 'water jug', 'water tower', 'whiskey jug', 'whistle', 'wig', 'window screen', 'window shade', 'Windsor tie', 'wine bottle', 'wing', 'wok', 'wooden spoon', 'wool, woolen, woollen', 'worm fence, snake fence, snake-rail fence, Virginia fence', 'wreck', 'yawl', 'yurt', 'web site, website, internet site, site', 'comic book', 'crossword puzzle, crossword', 'street sign', 'traffic light, traffic signal, stoplight', 'book jacket, dust cover, dust jacket, dust wrapper', 'menu', 'plate', 'guacamole', 'consomme', 'hot pot, hotpot', 'trifle', 'ice cream, icecream', 'ice lolly, lolly, lollipop, popsicle', 'French loaf', 'bagel, beigel', 'pretzel', 'cheeseburger', 'hotdog, hot dog, red hot', 'mashed potato', 'head cabbage', 'broccoli', 'cauliflower', 'zucchini, courgette', 'spaghetti squash', 'acorn squash', 'butternut squash', 'cucumber, cuke', 'artichoke, globe artichoke', 'bell pepper', 'cardoon', 'mushroom', 'Granny Smith', 'strawberry', 'orange', 'lemon', 'fig', 'pineapple, ananas', 'banana', 'jackfruit, jak, jack', 'custard apple', 'pomegranate', 'hay', 'carbonara', 'chocolate sauce, chocolate syrup', 'dough', 'meat loaf, meatloaf', 'pizza, pizza pie', 'potpie', 'burrito', 'red wine', 'espresso', 'cup', 'eggnog', 'alp', 'bubble', 'cliff, drop, drop-off', 'coral reef', 'geyser', 'lakeside, lakeshore', 'promontory, headland, head, foreland', 'sandbar, sand bar', 'seashore, coast, seacoast, sea-coast', 'valley, vale', 'volcano', 'ballplayer, baseball player', 'groom, bridegroom', 'scuba diver', 'rapeseed', 'daisy', "yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum", 'corn', 'acorn', 'hip, rose hip, rosehip', 'buckeye, horse chestnut, conker', 'coral fungus', 'agaric', 'gyromitra', 'stinkhorn, carrion fungus', 'earthstar', 'hen-of-the-woods, hen of the woods, Polyporus frondosus, Grifola frondosa', 'bolete', 'ear, spike, capitulum', 'toilet tissue, toilet paper, bathroom tissue']
            if args.dataset.lower() == "imagenet":
                if args.train_transform:
                    dataset1 = datasets.ImageFolder(root=os.path.join(args.data_path, "train"), transform=train_transform)
                else:
                    dataset1 = datasets.ImageFolder(root=os.path.join(args.data_path, "train"), transform=test_transform)
                dataset2 = datasets.ImageFolder(root=os.path.join(args.data_path, "val"), transform=test_transform)
            elif  "webvision" in args.dataset.lower() and "mini" in args.dataset.lower():
                args.num_classes = 50
                args.class_label_names = args.class_label_names[:50]
                if args.train_transform:
                    dataset1 = datasets.ImageFolder(root=os.path.join(args.data_path, "train"), transform=train_transform)
                else:
                    dataset1 = datasets.ImageFolder(root=os.path.join(args.data_path, "train"), transform=test_transform)
                dataset2 = datasets.ImageFolder(root=os.path.join(args.data_path, "val"), transform=test_transform)
            elif "webvision" in args.dataset.lower():
                if args.train_transform:
                    dataset1 = datasets.ImageFolder(root=os.path.join(args.data_path, "train"), transform=train_transform)
                else:
                    dataset1 = datasets.ImageFolder(root=os.path.join(args.data_path, "train"), transform=test_transform)
                dataset2 = datasets.ImageFolder(root=os.path.join(args.data_path, "val"), transform=test_transform)
            else:
                raise ValueError
    else:
        raise ValueError
    return dataset1, dataset2

def get_mislabeled_dataset(dataset, percentage_mislabeled, num_classes, return_clean_partition, path):
    if os.path.exists(f"{path}.mislabeled_points"):
        # load the mislabeled targets
        (index_list, old_targets, updated_targets) = torch.load(f"{path}.mislabeled_points") 
    else:
        if percentage_mislabeled == 0:
            return dataset, None, (None, None, None)
        # Generate and save the mislabeled targets
        num_of_data_points = len(dataset)
        num_of_mislabeled = int(percentage_mislabeled * num_of_data_points)
        r=np.arange(num_of_data_points)
        np.random.shuffle(r)
        index_list = r[:num_of_mislabeled].tolist()
        updated_targets = []
        old_targets = []
        for sample_index in index_list:
            if (torch.is_tensor(dataset.targets[sample_index])):
                old_targets.append(dataset.targets[sample_index].item())
                updated_targets.append(random.choice([val for val in range(num_classes) if val != dataset.targets[sample_index].item()]))
            elif isinstance(dataset.targets[sample_index], int ):
                old_targets.append(dataset.targets[sample_index])
                updated_targets.append(random.choice([val for val in range(num_classes) if val != dataset.targets[sample_index]]))            
        torch.save((index_list, old_targets, updated_targets), f"{path}.mislabeled_points")
    #Update the targets
    corrupt_samples = []
    for sample_number, sample_index in enumerate(index_list):
        if (torch.is_tensor(dataset.targets[sample_index])):
            dataset.targets[sample_index] = torch.ones_like(dataset.targets[0]) * updated_targets[sample_number]
            corrupt_samples.append(dataset[sample_index][0])
        elif isinstance(dataset.targets[sample_index], int ):
            dataset.targets[sample_index] = int(updated_targets[sample_number] )
            corrupt_samples.append(dataset[sample_index][0])

    if corrupt_samples:
        corrupt_samples = torch.stack(corrupt_samples)
        if len(corrupt_samples.shape)<4:
            corrupt_samples = corrupt_samples.unsqueeze(1)
    if return_clean_partition:
        mask = np.ones(len(dataset), dtype=bool)
        mask[index_list] = False
        dataset.data  = dataset.data[mask]
        dataset.targets = torch.tensor(dataset.targets)[mask]
    return dataset, corrupt_samples, (index_list, old_targets, updated_targets)

class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.v_label = torch.tensor ([0 for _ in range(len(dataset))]).long()
        self.indices = torch.tensor ([i for i in range(len(dataset))]).long()

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, self.v_label[index], self.indices[index]

    def __len__(self):
        return len(self.dataset)

def get_model(args,device):
    if  ( ("linear" in args.arch.lower() or "lenet" in args.arch.lower()) and ("mnist" not  in args.dataset.lower() ) ) or \
    ( ("vgg" in args.arch.lower() or "resnet" in args.arch.lower()) and ("mnist" in args.dataset.lower()) ) \
    or ("vit" in args.arch.lower() and  ( "imagenet" not in args.dataset.lower() or "webvision" not in args.dataset.lower() ) ):
        # line 1 -> Linear or LeNet arch -> MNIST Dataset
        # line 2 -> VGG or ResNet arch -> Not MNIST Dataset
        # line 3 -> VIT ->  imagenet or webvision dataset
        raise ValueError
    # Instantiate model
    if "linear" in args.arch.lower():
        model = Linear5(bias="nobias" not in args.arch.lower()).to(device)
    elif "lenet" in args.arch.lower():
        model = LeNet5(bias="nobias" not in args.arch.lower()).to(device)
    elif "vgg11_bn" in args.arch.lower():
        model = vgg11_bn(num_classes=args.num_classes, dataset=args.dataset.lower()).to(device)
    elif "vgg13_bn" in args.arch.lower():
        model = vgg13_bn(num_classes=args.num_classes, dataset=args.dataset.lower()).to(device)
    elif "vgg16_bn" in args.arch.lower():
        model = vgg16_bn(num_classes=args.num_classes, dataset=args.dataset.lower()).to(device)
    elif "vgg19_bn" in args.arch.lower():
        model = vgg19_bn(num_classes=args.num_classes, dataset=args.dataset.lower()).to(device)
    elif "vgg11" in args.arch.lower():
        model = vgg11(num_classes=args.num_classes, dataset=args.dataset.lower()).to(device)
    elif "vgg13" in args.arch.lower():
        model = vgg13(num_classes=args.num_classes, dataset=args.dataset.lower()).to(device)
    elif "vgg16" in args.arch.lower():
        model = vgg16(num_classes=args.num_classes, dataset=args.dataset.lower()).to(device)
    elif "vgg19" in args.arch.lower():
        model = vgg19(num_classes=args.num_classes, dataset=args.dataset.lower()).to(device)
    elif "resnet152" in args.arch.lower():
        model = ResNet152(num_classes=args.num_classes, dataset=args.dataset.lower()).to(device)
    elif "resnet101" in args.arch.lower():
        model = ResNet101(num_classes=args.num_classes, dataset=args.dataset.lower()).to(device)
    elif "resnet50" in args.arch.lower():
        model = ResNet50(num_classes=args.num_classes, dataset=args.dataset.lower()).to(device)
    elif "resnet34" in args.arch.lower():
        model = ResNet34(num_classes=args.num_classes, dataset=args.dataset.lower()).to(device)
    elif "resnet18" in args.arch.lower():
        model = ResNet18(num_classes=args.num_classes,dataset=args.dataset.lower()).to(device)  
    elif "inceptionresnetv2" in args.arch.lower():
        model = inceptionresnetv2(num_classes=args.num_classes, pretrained=False).to(device)
    elif "vit_b_16" in args.arch.lower():
        model = vit_b_16().to(device)
    elif "vit_b_32" in args.arch.lower():
        model = vit_b_32().to(device)
    elif "vit_l_16" in args.arch.lower():
        model = vit_l_16().to(device)
    elif "vit_l_32" in args.arch.lower():
        model = vit_l_32().to(device)
    elif "vit_h_14" in args.arch.lower():
        model = vit_h_14().to(device)
    else:
        raise ValueError
    return model

# SAM optimizer taken from https://github.com/davda54/sam/blob/main/sam.py
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

# GLS loss function taken from https://github.com/UCSC-REAL/negative-label-smoothing
def loss_gls(logits, labels, smooth_rate=0.1):
    # logits: model prediction logits before the soft-max, with size [batch_size, classes]
    # labels: the (noisy) labels for evaluation, with size [batch_size]
    # smooth_rate: could go either positive or negative, 
    # smooth_rate candidates we adopted in the paper: [0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6, -0.8, -1.0, -2.0, -4.0, -6.0, -8.0].
    confidence = 1. - smooth_rate
    logprobs = F.log_softmax(logits, dim=-1)
    nll_loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(1))
    nll_loss = nll_loss.squeeze(1)
    smooth_loss = -logprobs.mean(dim=-1)
    loss = confidence * nll_loss + smooth_rate * smooth_loss
    loss_numpy = loss.data.cpu().numpy()
    num_batch = len(loss_numpy)
    return torch.sum(loss)/num_batch


# Early stopper taken from https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# taken from https://github.com/LJY-HY/MentorMix_pytorch/blob/master/train_MentorNet.py
def MentorMixLoss(args, device, MentorNet, StudentNet, x_i, y_i,v_true, loss_p_prev, loss_p_second_prev, epoch):
    '''
    v_true is set to 0s in this version.
    inputs : 
        x_i         [bsz,C,H,W]
        outputs_i   [bsz,num_class]
        y_i         [bsz]
    intermediate :
        x_j         [bsz,C,H,W]
        outputs_j   [bsz,num_class]
        y_j         [bsz]
    outputs:
        loss        [float]
        gamma       [float]

    Simple threshold function is used as MentorNet in this repository.
    '''
    XLoss = torch.nn.CrossEntropyLoss(reduction='none')
    # MentorNet 1
    bsz = x_i.shape[0]
    x_i, y_i,v_true = x_i.to(device), y_i.to(device), v_true.to(device)
    with torch.no_grad():
        outputs_i = StudentNet(x_i) 
        loss = XLoss(outputs_i,y_i)                      
        loss_p = args.mnet_ema*loss_p_prev + (1-args.mnet_ema)*sorted(loss)[int(bsz*args.mnet_gamma_p)]
        loss_diff = loss-loss_p
        v = MentorNet(v_true,args.epochs, epoch,loss,loss_diff)   

        # Burn-in Process( Do only for few epoch)
        if epoch < min(args.mnet_burnin, args.epochs*0.2):
            v = torch.bernoulli(torch.ones_like(loss_diff)/2).to(device)

    P_v = cat.Categorical(F.softmax(v,dim=0))           
    indices_j = P_v.sample(y_i.shape)                   
    
    # Prepare Mixup
    x_j = x_i[indices_j]
    y_j = y_i[indices_j]
    
    # MIXUP
    Beta = diri.Dirichlet(torch.tensor([args.mmix_alpha for _ in range(2)]))
    lambdas = Beta.sample(y_i.shape).to(device)
    lambdas_max = lambdas.max(dim=1)[0]                 
    lambdas = v*lambdas_max + (1-v)*(1-lambdas_max)     
    x_tilde = x_i * lambdas.view(lambdas.size(0),1,1,1) + x_j * (1-lambdas).view(lambdas.size(0),1,1,1)
    outputs_tilde = StudentNet(x_tilde)
    
    # Second Reweight
    with torch.no_grad():
        loss = lambdas*XLoss(outputs_tilde,y_i) + (1-lambdas)*XLoss(outputs_tilde,y_j)
        loss_p_second = args.mnet_ema*loss_p_second_prev + (1-args.mnet_ema)*sorted(loss)[int(bsz*args.mnet_gamma_p)]
        loss_diff = loss-loss_p_second
        v_mix = MentorNet(v_true,args.epochs, epoch,loss,loss_diff)

        # Burn-in Process( Do only for few epoch)
        if epoch < min(args.mnet_burnin, args.epochs*0.2):
            v_mix = torch.bernoulli(torch.ones_like(loss_diff)/2).to(device)

    loss = lambdas*XLoss(outputs_tilde,y_i) + (1-lambdas)*XLoss(outputs_tilde,y_j)
    masked_loss = loss*v_mix
  
    return XLoss(outputs_i,y_i).mean(), masked_loss.mean(), loss_p, loss_p_second, v

# taken from https://github.com/LJY-HY/MentorMix_pytorch/blob/master/train_MentorNet.py
def MentorNetLoss(args, device, MentorNet, StudentNet, x_i, y_i,v_true, loss_p_prev,  epoch):
    '''
    v_true is set to 0s in this version.
    inputs : 
        x_i         [bsz,C,H,W]
        outputs_i   [bsz,num_class]
        y_i         [bsz]
    intermediate :
        x_j         [bsz,C,H,W]
        outputs_j   [bsz,num_class]
        y_j         [bsz]
    outputs:
        loss        [float]
        gamma       [float]

    Simple threshold function is used as MentorNet in this repository.
    '''
    XLoss = torch.nn.CrossEntropyLoss(reduction='none')
    # MentorNet 1
    bsz = x_i.shape[0]
    x_i, y_i,v_true = x_i.to(device), y_i.to(device), v_true.to(device)
    with torch.no_grad():
        outputs_i = StudentNet(x_i) 
        loss = XLoss(outputs_i,y_i)                      
        loss_p = args.mnet_ema*loss_p_prev + (1-args.mnet_ema)*sorted(loss)[int(bsz*args.mnet_gamma_p)]
        loss_diff = loss-loss_p
        v = MentorNet(v_true,args.epochs, epoch,loss,loss_diff)           
        
        # Burn-in Process( Do only for few epoch)
        if epoch < min(args.mnet_burnin, args.epochs*0.2):
            v = torch.bernoulli(torch.ones_like(loss_diff)/2).to(device)

    output = StudentNet(x_i)
    loss = XLoss(output,y_i)
    masked_loss = loss*v
    return loss.mean(), masked_loss.mean(), loss_p, v
    