import csv
import os

def csv_dataset(path):
    folder_list = os.listdir(path)
    with open(os.getcwd()+"/datasets/data.csv", "a+") as outfile:
        #writer = csv.writer(outfile)
        #writer.writerow(['Label', 'Email', 'Message'])
        for f in folder_list:
            file_list = os.listdir(os.path.join(path, f))
            for file in file_list:
                with open(os.path.join(path, f, file), "r")  as infile:
                    contents = infile.read()
                    outfile.write(f+',')
                    outfile.write(contents)
    return (os.getcwd()+"/datasets/data.csv")
