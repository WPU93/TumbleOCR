import json
lines = open("en_dict/en_keys.txt").readlines()
wf = open("en_dict/en_dict.txt","w")
idf = open("en_dict/en_id2char.json","w")
charf = open("en_dict/en_char2id.json","w")

char2id = {}
id2char = {}
for id,char in enumerate(lines):
    char = char.strip("\n").split(" ")[0]
    char2id[char] = id+1
    id2char[id+1] = char
    wf.write(char+"\t"+str(id+1)+"\n")
json.dump(char2id,charf)
json.dump(id2char,idf)
    

