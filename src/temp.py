
with open("../Data/netflix_data/combined_data_1.txt",'r') as f:
    user = []
    for line in f:
        if line[-2] == ":":
            if int(line.split(":")[0]) == 31:
                break
            continue
        id = int(line.split(",")[0])
        user.append(id)
    print(len(list(set(user))))


