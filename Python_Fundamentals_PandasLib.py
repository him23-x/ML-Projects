import pandas as pd


### Sreies ###


#data = [100,150,200,250,300,400]

# Series = pd.Series(data)
# print(Series)
# Series = pd.Series(data, index = ["a","b","c","d","e","f"])

# print(Series)
# Series.loc["b"] = 500
# print(Series.loc["b"])# Search by label
# print(Series.iloc[2]) # Search by Index
# print(Series[Series > 200])


# caluries = {
#     "Day1" : 2730,
#     "Day2" : 2500,
#     "Day3" : 1900,
#     "Day4" : 1600,
# }

# Series = pd.Series(caluries)

# print(Series.loc["Day3"])

# Series.loc["Day2"] += 100
# print(Series)
# print(Series[Series > 2000])
















### Data Frames ###


# Employees = {
#     "Name" : ["Ebrahim", "Mohammed","Farouk"],
#     "Age" : [21, 11, 6]
# }

# df = pd.DataFrame(Employees, index = ["Em1", "Em2", "Em3"])

# print(df)
# print(df.loc["Em1"])
# print(df.iloc[1])


#Add Column
# df["Job"] = ["Software Engineer", "Scientist", "N/A"] #N/A meaning is Not A value
# print(df)

#Add Row

# NewRow = pd.DataFrame([{"Name" : "Maouda", "Age" : 15, "Job" : "Doctor"}],
#                       index = ["Em4"])

# df = pd.concat([df, NewRow])
# print(df)



#Add Rows

# NewRows = pd.DataFrame([{"Name" : "Maouda", "Age" : 15, "Job" : "Doctor"},
#                       {"Name" : "Mahmoud", "Age" : 17, "Job" : "N/A"}],
#                       index = ["Em4", "Em5"])

# df = pd.concat([df, NewRows])

# print(df)











### Importing ###

# df = pd.read_csv("data.csv")
# print(df)
# print(df.to_string())










### Selection ###

##By Column
# df = pd.read_csv("data.csv")

# print(df["Name"])
# print(df[["Name", "Height", "Weight"]])



##By Row
# df = pd.read_csv("data.csv")
# df = pd.read_csv("data.csv", index_col="Name") # its make Name is and unique identifier, 
# and we can search by Name Columns

# print(df.iloc[2])
# print(df.loc[2,["Name","Height"]])Squirtle
# print(df.loc[2:10,["Name","Height"]])

# print(df.iloc[0:11:2, 0:3]) #df.iloc[slice rows, slice columns]
#remember that: the last number is execlusive.


# pichomon = input("Enter pichomon Name: ")


# try :
#     print(df.loc[pichomon])
# except KeyError:
#     print(f"{pichomon} is not Found!!!")











###Filtering###

# df = pd.read_csv("data.csv")

# Tall = df[df["Height"] >= 2]
# water_pochomon = df[(df["Type1"] == "Water") | (df["Type2"] == "Water")]
# ff_pochomon = df[(df["Type1"] == "Fire") & (df["Type2"] == "Flying")]

# print(ff_pochomon)














###Aggregate Function###

# df = pd.read_csv("data.csv")


# Whole Data Frame
# print(df.mean(numeric_only=True))
# print(df.sum(numeric_only=True))
# print(df.max(numeric_only=True))
# print(df.min(numeric_only=True))
# print(df.count())



#Single Column
# print(df["Height"].mean(numeric_only=True))
# print(df["Height"].sum(numeric_only=True))
# print(df["Height"].max(numeric_only=True))
# print(df["Height"].min(numeric_only=True))
# print(df["Height"].count())



#Grouped By

# group = df.groupby("Type1")

# print(group["Height"].mean())




















#Data Cleaning
df = pd.read_csv("data.csv")

#1. Drop Irrelevant Columns

# df = df.drop(columns=["Legendary", "No"])
# print(df)

#2. Handling a missing data

# df = df.dropna(subset = ["Type2"]) # will remove any row missing a data

# df = df.fillna({"Type2": "None"})
# print(df.to_string())


# 3. Fix inconsistent Values

# df["Type1"] = df["Type1"].replace({"Grass": "GRASS",
#                                    "Fire" : "FIRE"})
# print(df.to_string())


# 4. Standrize text

# df["Name"] = df["Name"].str.upper()
# print(df.to_string())


# 5. Fix data types

# df["Legendary"] = df["Legendary"].astype(bool)
# print(df.to_string())

# 6. Remove duplicates data

# df = df.drop_duplicates()
# print(df.to_string())