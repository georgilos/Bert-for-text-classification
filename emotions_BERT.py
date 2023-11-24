from transformers import TFAutoModel, AutoTokenizer
from datasets import load_dataset
import tensorflow as tf



model = TFAutoModel.from_pretrained("bert-base-uncased")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

inputs = tokenizer(['Hello world', 'Hi how are you'], padding=True # adds 0's where tokens have no meaning, making lists the same length
                  , truncation=True #trancates any text with > 512 words
                   ,return_tensors='tf') #converts from list to tf tensor

#print(inputs)

output = model(inputs) #feeding tokens to bert

#print(output)

emotions = load_dataset('SetFit/emotion') #dataset doanload with train, test, validation sets with 1600, 2000, 2000 samples each

#print(emotions)

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True) #tokenize func to tokenize every text/split in dataset

emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None) #new dataset with input_ids,token_type_ids,attention_mask

#print(emotions_encoded)

#convertion of data set from hugging face format to tensor flow data set format:

emotions_encoded.set_format('tf',columns=['label', 'input_ids', 'token_type_ids', 'attention_mask']) # setting 'input_ids', 'attention_mask','token_type_ids', and 'label' colums to the tensorflow format


BATCH_SIZE = 64 # setting BATCH_SIZE to 64

def order(inp):

#     grouping the inputs of BERT into a single dictionary and then output it with labels.

    data = list(inp.values())
    return {
        'input_ids': data[1], 
        'token_type_ids': data[2],
        'attention_mask': data[3] #inputs for bert
    }, data[0] #output labels

# converting train split of `emotions_encoded` to tensorflow format
train_dataset = tf.data.Dataset.from_tensor_slices(emotions_encoded['train'][:])
# set batch_size and shuffle
train_dataset = train_dataset.batch(BATCH_SIZE).shuffle(1000)
# map the `order` function
train_dataset = train_dataset.map(order, num_parallel_calls=tf.data.AUTOTUNE)

#doing the same for test set
test_dataset = tf.data.Dataset.from_tensor_slices(emotions_encoded['test'][:])
test_dataset = test_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.map(order, num_parallel_calls=tf.data.AUTOTUNE)

inp, out = next(iter(train_dataset)) # a batch from train_dataset
#print(inp, '\n\n', out)

class BERTForClassification(tf.keras.Model): #using subclassing api in keras by inheriting tf.keras.model class in the init func
    
    def __init__(self, bert_model, num_classes):
         super().__init__() #calling the init func of parent class to initialise model
         self.bert = bert_model
         self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')
        
    def call(self, inputs): 
        x = self.bert(inputs)[1] #pooler output needed for text classification
        return self.fc(x) #returning last dense layer
    
classifier = BERTForClassification(model, num_classes=6) #creating an instance if our model

classifier.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), #slow learing rate to avoid overfitting
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# history = classifier.fit(
#     train_dataset,
#     epochs=3
# )



#classifier.evaluate(test_dataset)