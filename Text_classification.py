def Text_Prepressing(df):
    #Information about the dataset
    print("Some Basic Description about the dataset:")
    print(df.info())
    print("-"*50)
    print(df.describe())
    print("-"*50)
    #some EDAs
    #Checking distribuition of two classes
    import matplotlib.pyplot as plt
    plt.hist(df['sentiment'])
    print("Checking the distribution of two classes:")
    plt.show()
    print("-"*50)

    #Checking null values
    print("Checking null values:")
    print(df['sentiment'].isnull().sum())
    print("-"*50)

    #Duplicate values
    print("Checking duplicate values:")
    duplicate = df.duplicated().sum()
    if df.duplicated().sum() > 0:
        df.drop_duplicates(inplace=True)
        print(f"{duplicate} Duplicate values found and removed.")
        print("-"*50)
    else:
        print("-"*50)

    #Removing html tags
    import re

    def remove_html_tags(text):
        if isinstance(text, str):
            return re.sub(r'<.*?>', '', text)
        else:
            return text
    df['review'] = df.iloc[:,0].apply(remove_html_tags)
    print("HTML tags removed.")
    print("-"*50)

    #Chat word treatment
    from Chat_treatment import chat_dict
    def replace_chat_words(text):
        for word in text.split():
            if word.upper() in chat_dict:
                text = text.replace(word, chat_dict[word.upper()])
        return text
    df['review'] = df['review'].apply(replace_chat_words)
    print("Chat words replaced.")
    print("-"*50)

    #Lowercasing
    df['review'] = df['review'].str.lower()
    print("Lowercasing done.")
    print("-"*50)

    #Removing punctuation
    import string
    def remove_pun(text):
        return (text.translate(str.maketrans('', '', string.punctuation)))
    df['review'] = df['review'].apply(remove_pun)
    print("Punctuation removed.")
    print("-"*50)

    #Stopword removal
    import nltk 
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    def remove_stopwords(text):
        new_text = []
        for word in text.split():
            if word not in stop_words:
                new_text.append(word)
        return " ".join(new_text)
    df['review'] = df['review'].apply(remove_stopwords)
    print("Stopwords removed.")
    print("-"*50)
    return(df)
