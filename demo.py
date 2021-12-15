import csv

csv_output = []
with open('models/sample_29.txt', 'r') as file:
    lines = [line.split() for line in file]
    j = 0
    for i in range(6, len(lines), 6):
        question = lines[j:i]
        j = i
        correct_idx = ord(question[5][0]) - ord('a')
        csv_output.append([[question[0][1], question[1+correct_idx][1], question[1][1], question[2][1], question[3][1], question[4][1]]])
with open('models/sample_29.csv', 'w') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerows([['question', 'answer','0','1','2','3']])
    for line in csv_output:
        writer.writerows(line)

    