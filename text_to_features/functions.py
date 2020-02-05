import os

reviews = list()

# negative_polarity directory
# ---------------------------------------------------------
files_in_directory_negative_polarity = os.listdir("../datasets/op_spam_v1.4/test_op_spam_v1.4/negative_polarity/")
file_path_negative_polarity = "../datasets/op_spam_v1.4/test_op_spam_v1.4/negative_polarity/"

# Loop over the files in negative_polarity directory
# Open the file line by line
for file_name in files_in_directory_negative_polarity:
    print("The file is:: ", file_name)
    print("File flag:: ", file_name[0])
    file_flag = file_name[0]
    file_path = file_path_negative_polarity + file_name
    file_open = open(file_path)
    review = file_open.readline()
    print("The Review is:: ")
    print(review)
    reviews.append(review)


# positive_polarity directory
# ---------------------------------------------------------
files_in_directory_positive_polarity = os.listdir("../datasets/op_spam_v1.4/test_op_spam_v1.4/positive_polarity/")
file_path_positive_polarity = "../datasets/op_spam_v1.4/test_op_spam_v1.4/positive_polarity/"

# Loop over the files in positive_polarity directory
# Open the file line by line
for file_name in files_in_directory_positive_polarity:
    print("The file is:: ", file_name)
    print("File flag:: ", file_name[0])
    file_flag = file_name[0]
    file_path = file_path_positive_polarity + file_name
    file_open = open(file_path)
    review = file_open.readline()
    print("The Review is:: ")
    print(review)
    reviews.append(review)


print(reviews)
