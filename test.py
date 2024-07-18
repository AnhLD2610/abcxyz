# def _read_descriptions(file):
#         # id2rel, rel2id = {}, {}
#         rel2des = {}
#         id2des = {}
#         with open(file) as f:
#             for index, line in enumerate(f):
#                 rel = line.strip()
#                 x = rel.split('\t')
#                 rel2des[x[1]] = x[2]
#                 id2des[int(x[0])] = x[2]
#         return rel2des, id2des 

# _read_descriptions('/media/data/thanhnb/Bi/abc_cpl/sadfasdfasdf/data/CFRLFewRel/relation_description.txt')


# Example of a dictionary of dictionaries
data = {
    'item1': {
        'name': 'Item One',
        'price': 100,
        'quantity': 5
    },
    'item2': {
        'name': 'Item Two',
        'price': 200,
        'quantity': 3
    },
    'item3': {
        'name': 'Item Three',
        'price': 150,
        'quantity': 10
    }
}

# Print the dictionary of dictionaries
print(data)

# Accessing elements in the dictionary of dictionaries
item1_price = data['item1']['price']
print(f"The price of item1 is {item1_price}")
