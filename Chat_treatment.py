#Initialise empty dictionary
chat_dict = {}
#Defining delimiter
delimiter = '='
#Reading file
with open('slang.txt','r') as file:
    for line in file:
        #Splitting each line into key and value
        if delimiter in line:
            key, value = line.split(delimiter, 1)
            chat_dict[key.strip()] = value.strip()
