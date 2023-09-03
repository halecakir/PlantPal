
import string
out = open("tagged_slots_out.txt", "w")
with open("tagged_slots.txt") as f:
    sentences = []
    sent = None
    intent = None
    tags = {}
    try:
        for line in f:
            line = line.strip()
            if line:
                if ";" in line:
                    sent, intent = line.split(";")
                    sent  = sent.translate(str.maketrans('', '', string.punctuation))
                    sentences.append([sent, intent])
                else:
                    tag, word = line.split(":")
                    word = word.lstrip().rstrip()
                    tags[word] = tag
            else:
                if sent:
                    out.write(sent + ";" + intent + ";")
                    for word in sent.split():
                        found = False
                        
                        if word in tags:
                            out.write(tags[word] + " ")
                        else:
                            out.write("O ") 
                    out.write("\n")
                    sent = None
                    intent = None
                    tags = {}
    except Exception as e:
        print(line, sent)