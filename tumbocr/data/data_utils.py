import os

def get_vocabulary(path):
    char2id = {}
    id2char = {}
    with open(path,"r",encoding="utf-8-sig") as rf:
        lines = rf.readlines()
        for line in lines:
            char,id = line.strip("\n").split("\t")
            id2char[int(id)] = char
            char2id[char] = int(id)
    return char2id, id2char
