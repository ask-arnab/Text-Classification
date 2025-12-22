class TextPreprocessing:
    def __init__(self,df):
        self.df = df
        
    def Sample(self,number):
        self.df = self.df.sample(10000, random_state=42)
        
    def BasicDescription(self):
        print("Some Basic Description about the dataset:")
        print(self.df.info())
        print("-"*50)
        print(self.df.describe())
        print("-"*50)
        
    def EDA(self):
        import matplotlib.pyplot as plt
        plt.hist(self.df.iloc[:,1])
        print("Checking the distribution of two classes:")
        plt.show()
        print("-"*50)
        
    def Null_Values(self):
        print("Checking null values:")
        print(self.df.iloc[:,1].isnull().sum())
        print("-"*50)
        
    def Remove_Duplicates(self):
        print("Checking duplicate values:")
        duplicate = self.df.duplicated().sum()
        if self.df.duplicated().sum() > 0:
            self.df.drop_duplicates(inplace=True)
            print(f"{duplicate} Duplicate values found and removed.")
            print("-"*50)
        else:
            print("-"*50)
            
    def Remove_HTML(self):
        import re

        def remove_html_tags(text):
            if isinstance(text, str):
                return re.sub(r'<.*?>', '', text)
            else:
                return text
            
        self.df['review'] = self.df.iloc[:,0].apply(remove_html_tags)
        print("HTML tags removed.")
        print("-"*50)
    
    def Chat_Treatment(self):
        from Chat_treatment import chat_dict
        def replace_chat_words(text):
            for word in text.split():
                if word.upper() in chat_dict:
                    text = text.replace(word, chat_dict[word.upper()])
            return text
        self.df['review'] = self.df['review'].apply(replace_chat_words)
        print("Chat words replaced.")
        print("-"*50)
        
    def Lowercasing(self):
        self.df['review'] = self.df['review'].str.lower()
        print("Lowercasing done.")
        print("-"*50)
        
    def Punctuation(self):
        import string
        def remove_pun(text):
            return (text.translate(str.maketrans('', '', string.punctuation)))
        
        self.df['review'] = self.df['review'].apply(remove_pun)
        print("Punctuation removed.")
        print("-"*50)
        
    def Stopwords(self):
        import nltk 
        from nltk.corpus import stopwords
        stop_words = stopwords.words('english')
        def remove_stopwords(text):
            new_text = []
            for word in text.split():
                if word not in stop_words:
                    new_text.append(word)
            return " ".join(new_text)
        
        self.df['review'] = self.df['review'].apply(remove_stopwords)
        print("Stopwords removed.")
        print("-"*50)
        
    def All(self):
        self.BasicDescription()
        self.EDA()
        self.Null_Values()
        self.Remove_Duplicates()
        self.Remove_HTML()
        self.Chat_Treatment()
        self.Lowercasing()
        self.Punctuation()
        self.Stopwords()
        
    def export(self):
        return (self.df.to_csv("Cleaned.csv",index=False))
        
import pandas as pd
df = pd.read_csv("IMDB Dataset.csv")

TP = TextPreprocessing(df)
TP.Lowercasing()
