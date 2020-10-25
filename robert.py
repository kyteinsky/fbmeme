# Load the model in fairseq
from fairseq.models.roberta import RobertaModel
from spell_ck import spelling


class roberta_enc():
    def __init__(self, model_path:str='./roberta.large', spell:bool=True):
        """
        spell: Use spell checker
        """
        self.roberta = RobertaModel.from_pretrained(model_path, checkpoint_file='model.pt')
        #self.roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
        self.roberta.eval()
        self.sp = spelling()


    def get_features(self, text:str):
        """ 
        1. Check spelling and correct them.
        2. Tokenize the corpus
        3. Forward pass and get feature vector
        <Truncated to 100 tokens MAX>
        """
        text = self.sp.check(text)
        print(text)
        tokens = self.roberta.encode(text)
        tokens = tokens[:100]

        return self.roberta.extract_features(tokens)



# if __name__ == '__main__':
#     rob = roberta_enc()
#     text = '''Brousing all of the model's parameters by name here. In the below cell, we have names and dimensions of the weights for: The embedding layer. The first of the twelve transformers. The output layer'''
    
#     features = rob.get_features(text)

#     print('features size =',features.size())

