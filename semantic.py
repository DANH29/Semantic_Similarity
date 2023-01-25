import spacy

nlp = spacy.load('en_core_web_md')

# running "cat", "monkey" and "banana"

word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print(f"{word1}, {word2}: {word1.similarity(word2)}")
print(f"{word3}, {word2}: {word3.similarity(word2)}")
print(f"{word3}, {word1}: {word3.similarity(word1)}\n")

# cat and monkey have the highest similarity as they're both animals
# it is interesting that bananas have double the similarity with monkeys than cats
# this is because monkeys eat bananas

# creating my own example to compare similarity between words

word4 = nlp("cats")
word5 = nlp("dogs")
word6 = nlp("rain")

print(f"{word4}, {word5}: {word4.similarity(word5)}")
print(f"{word4}, {word6}: {word4.similarity(word6)}")
print(f"{word5}, {word6}: {word5.similarity(word6)}\n")

# cats and dogs have the highest similarity as they're both animals
# cats and dogs have almost equal similarity with rain
# this is due to the phrase: "Raining cats and dogs"

# running working with vectors example

tokens = nlp('cat apple monkey banana ')

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

print("")

# running working with sentences example

sentence_to_compare = "Why is my cat on the car"

sentences = ["Where did my dog go",
             "Hello, there is my car",
             "I\'ve lost my car in my car",
             "I\d like my boat back",
             "I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)


# running example.py using en_core_web_sm instead of en_core_web_md

# when I run example.py with en_core_web_sm I get a UserWarning [W007]
# which informs me that the model I am using has no word vectors loaded
# without vectors my program may not give useful similarity judgements
