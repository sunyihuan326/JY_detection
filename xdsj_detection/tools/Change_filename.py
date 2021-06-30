import os

files = os.listdir(r"C:\Users\bai\Desktop\abc")
for file_name in files:
    print(file_name)

    portion = os.path.splitext(file_name)
    print(portion)

    if portion[1] == ".png":
        new_name = portion[0] + ".jpg"
        os.rename(file_name, new_name)