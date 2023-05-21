# Import required libraries
import numpy as np
import torch
import torch.nn.functional as F
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from lime.lime_text import LimeTextExplainer

def load_model(model_path):
    # Load the tokenizer and model from the "roberta-base" pre-trained model
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained("roberta-base").cuda()

    # Load the saved state dict of the fine-tuned model
    model.load_state_dict(torch.load(model_path))

    return tokenizer, model

# Load the fine-tuned model from the saved state dict
model_path = "best_model_roberta_mixed.pt"
tokenizer, model = load_model(model_path)
model = model.to('cpu')

# define the explainer
class_names = ['human', 'chatGPT']
explainer = LimeTextExplainer(class_names=class_names)

def get_predict_probs(texts): 
    outputs = model(**tokenizer(texts, return_tensors="pt", padding=True))
    tensor_logits = outputs[0]
    # Get the predicted label using the input_ids and attention_mask
    probas = F.softmax(tensor_logits).detach().cpu().numpy()
    return probas

text1 = "Basically there are many categories of \" Best Seller \" . Replace \" Best Seller \" by something like \" Oscars \" and every \" best seller \" book is basically an \" oscar - winning \" book . May not have won the \" Best film \" , but even if you won the best director or best script , you 're still an \" oscar - winning \" film . Same thing for best sellers . Also , IIRC the rankings change every week or something like that . Some you might not be best seller one week , but you may be the next week . I guess even if you do n't stay there for long , you still achieved the status . Hence , # 1 best seller .", "If you 're hearing about it , it 's because it was a very good or very well - publicized book ( or both ) , and almost every good or well - publicized book will be # 1 on the NY Times bestseller list for at least a little bit . Kindof like how almost every big or good movies are # 1 at the box office on their opening weekend .", "One reason is lots of catagories . However , how the NY Times calculates its best seller list is n't comprehensive , and is pretty well understood by publishers . So publishers can [ buy a few books ] ( URL_0 ) in the right bookstores and send a book to the top of the list for at least a week ."
text2 = "There are many different best seller lists that are published by various organizations, and the New York Times is just one of them. The New York Times best seller list is a weekly list that ranks the best-selling books in the United States based on sales data from a number of different retailers. The list is published in the New York Times newspaper and is widely considered to be one of the most influential best seller lists in the book industry. \nIt's important to note that the New York Times best seller list is not the only best seller list out there, and there are many other lists that rank the top-selling books in different categories or in different countries. So it's possible that a book could be a best seller on one list but not on another. \nAdditionally, the term \"best seller\" is often used more broadly to refer to any book that is selling well, regardless of whether it is on a specific best seller list or not. So it's possible that you may hear about a book being a \"best seller\" even if it is not specifically ranked as a number one best seller on the New York Times list or any other list."

text3 = "salt is good for not dying in car crashes and car crashes are worse for cars then salt . Some places use other things , but salt is really cheap compared to most alternatives , although sand is pretty good .", "In Minnesota and North Dakota , they tend to use sand , not salt . In these states , though , it is cold enough that the salt can not actually melt the snow , and so it refreezes and you end up with black ice on the roads , which is counterproductive . In other states whether there is snow but not so cold , wet asphalt is better than gritty snow for traction and so salt is favored for safety . Most people who live in states where salt is used , and by the way also in coastal cities , know that washing the car a couple times a week is critical .", "Used to work in the salt industry ( yes , it 's a thing ) . Salt is one of the cheapest substances on earth . As in , you 'd be getting ripped off if you paid more than $ 150 per metric ton . ( In fact , the largest component of salt 's price is actually shipping . Yes , it costs more to transport salt than it does to produce it . ) Salt does its job ( lowering the freezing point of water ) incredibly effectively and is ludicrously abundant in nature . Its crystalline structure can also be manipulated in order to have sharper edges that reduce bounce . So that allows it to stick evenly to critical surfaces such as roads . Yes , there are alternatives . Usually salt / gravel mix or sugar beet based solutions . Those have less of an environmental impact when the spring melt washes it into the drain . This is more used in places where balance of water salinity for the safety of aquatic life is important . Propylene glycol , a water displacer , is even used in some places where it 's so cold that salt wo n't work effectively . But salt is by far the cheapest , most effective solution for preventing the icing of roads . And your government wants above all cheap and effective solutions that reduce winter accidents and keep people safe . They do n't care as much about the finish on your 1994 Honda Prelude ."
text4 = "Salt is used on roads to help melt ice and snow and improve traction during the winter months. When it's cold outside, water can freeze on the roads and make them very slippery, which can be dangerous for cars and people. Salt helps to melt the ice and snow by lowering the freezing point of water, which means that it can help keep the roads clear and safe to travel on. \nThere are other options for melting ice and snow on roads, such as using chemicals like calcium chloride or magnesium chloride, or using mechanical methods like plows or sand. However, salt is often the most effective and affordable option for many communities, especially when it's used in combination with other methods. \nIt's important to note that while salt can be helpful for making roads safer during the winter, it can also have negative effects on the environment and on the cars themselves. Salt can cause corrosion on metal surfaces, including cars, and it can also harm plants and animals if it washes into nearby waterways. However, despite these potential downsides, many communities continue to use salt as a way to keep roads clear and safe during the winter."

text1 = str(text1)
text2 = str(text2)
text3 = str(text3)
text4 = str(text4)

exp = explainer.explain_instance(text3, get_predict_probs, num_features=12, num_samples=200)

html_file = '../lime_visualization/lime_humanSample2.html'
exp.save_to_file(file_path=html_file)