import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv(r"C:\Users\abdel\Downloads\data\ProjectData.csv")
print(data.head())

print(data.info())




print(data.columns)


def fix_Marital(a):
    if(a=="M" or a=="m" or a=="married" or a=="Married"):
        return "married"
    else:
        return "single"
data["Marital Status"]=data["Marital Status"].apply(fix_Marital)

data[' Income '] = data[' Income '].str.replace('$', '', regex=False).str.replace(',', '', regex=False)



data[' Income '] = pd.to_numeric(data[' Income '], errors='coerce')

print(data.info())


print(data.describe(),"\n")

def fix_Children(a):

    if (a=="one"  or a=="1"):
        return "1"
    elif (a=="zero"  or a=="0"):
        return "0"
    elif (a=="two"  or a=="2"):
        return "2"
    elif (a=="three"  or a=="3"):
        return "3"
    elif (a=="four"  or a=="4"):
        return "4"
    elif (a=="five"  or a=="5"):
        return "5"
    elif (a=="six"  or a=="6"):
        return "6"
    else:
        return "0"
    
data['Children' ] = data['Children'].apply(fix_Children).astype(int)




def fix_PurchasedBike(a):
    if ("y" in a or "Y" in a or "Yes" in a or "yes" in a):
        
        return "1"
    else:
        return "0"
    
data['Purchased Bike'] = data['Purchased Bike'].apply(fix_PurchasedBike).astype(int)




def fix_home_owne(a):
    if ("y" in a or "Y" in a or "Yes" in a or "yes" in a):
        
        return "1"
    else:
        return "0"
    
data['Home Owner'] = data['Home Owner'].apply(fix_PurchasedBike).astype(int)


def fix_gender(a):
    if a=="M" or a=="m" or a=="male" or a=="Male":
        return "male"
    else :
        return "female"
data['Gender'] = data['Gender'].apply(fix_gender)


data=data.fillna(data.mode().iloc[0])
print("number of duplicated data ",data.duplicated().sum(),"\n")
data=data.drop_duplicates()
print(data.info())






# 1. What is the average income of the employees  in this dataset?
avg_income = data[' Income '].mean()
print("Average income: ",avg_income)


# 2. Which percentage of employees earn more than $50,000 ?
percentage=(data[' Income '] > 50000).sum() 


print("percentage of employees earn more than $50,000: ",percentage/len(data) *100,"% \n")




# 3. Which percentage of employees have purchased a bike?
percentage=(data['Purchased Bike']==1).sum() / len(data) * 100
print("percentage of employees have purchased a bike: ",percentage,"% \n")



# 4. What is the most common occupation in this dataset?
common_occupation = data['Occupation'].mode()
print("Most common occupation: ",common_occupation,"\n")



# 5. How many employees in this dataset have no children?
num_no_children = (data['Children'] == 0).sum()
print("Number of  with no children:",num_no_children,"\n")



# 6. What is the average number of cars owned by employeesls in this dataset?
avg_num_cars = data['Cars'].mean()
print("Average number of cars: ",avg_num_cars,"\n")



# 7. How many employees in this dataset live in the Pacific region?
num_pacific_region = (data['Region'] == 'Pacific').sum()
print("Number of employees live in  Pacific region: ",num_pacific_region,"\n")


# 8. What is the average age  in this dataset?
print("the average age :",data['Age'].mean(),"\n")


a=0

    
for i,n in zip(data["Home Owner"] , data["Commute Distance"]):
    if (n == '5-10 Miles') and (i == 1):
        a+=1

#9. what is the percentage in this dataset who have a commute distance of 5-10 miles and own a home?
print("the percentage in this dataset who have a commute distance of 5-10 miles and own a home: ",a/len(data["Home Owner"])*100,"%\n")


#10. What is the most common commute distance in this dataset?
print("the most common commute distance",data["Commute Distance"].mode().iloc[0],"\n") 


# 11. What is the most common gender in this dataset?

print("the most common commute gender:",data["Gender"].mode().iloc[0],"\n")

#12.what is the average income of male employees
male_income_mean = data[data['Gender']=='male'][' Income '].mean()
print("the average income of male individuals:",male_income_mean,"\n")


#13.what is the average income of female employees
female_income_mean = data[data['Gender']=='female'][' Income '].mean()
print("the average income of female individuals:",female_income_mean,"\n")


#14. What is the percentage of male employees in this dataset?
percentage_male = (data['Gender'] =="male").sum() / len(data) * 100
print("the percentage of male :",percentage_male,"% \n")


#15.What is the percentage of female employees in this dataset?
print("the percentage of female :",100-percentage_male,"% \n")

#16.How many employee in this dataset have 2 or more cars and income <50000?
print("employee in this dataset have 2 or more cars and income <50000: ",((data['Cars'] >= 2) & (data[' Income '] < 50000)).sum(),"employee\n")

#17.What is the percentage of indiviuals in this dataset who are home owners and have purchased a bike?
percentage=(((data["Home Owner"]==1)& (data["Purchased Bike"]==1)).sum())/len(data) *100
print("percentage of individuals in this dataset who are home owners and have purchased a bike : ",percentage,"%\n")

#18.What is the highest income in this dataset?
print("highest income: ",data[" Income "].max(),"\n")


#19.How many employee in this dataset have a partial college education?

print("partial college education: ",(data['Education']=="Partial College").sum(),"employee \n")

#20.How many employee in this dataset are over 50 years old?
employe=(data["Age"]>50).sum()
print("employee in this dataset are over 50 years old : ",employe," employee\n")
#21.What is the percentage of male employees in over 50 years old?
employe=((data["Age"]>50) & (data["Gender"]==1)).sum()
print(" the percentage of male employees in over 50 years old: ",employe,"employee \n")

#22.How many employee in this dataset have a skilled manual occupation?
employe=(data["Occupation"]=="Skilled Manual").sum()
print("employee in this dataset have a skilled manual occupation",employe,"employee \n")


#23.what is the ID of all rows with the highest income?
max_income = data[' Income '].max()
id = data.loc[data[' Income '] == max_income, 'ID']
print("id for the highest income: ",list(id),"\n")


#24.How many employee in this dataset have a graduate degree?
num_graduate = ((data['Education'] == 'Graduate') | (data['Education'] == 'Bachelors')).sum()
print("employee in this dataset have a graduate degree: ",num_graduate,"employee \n")
 
#25.what is the avrege income in Europe?
avg=data.loc[data['Region'] == 'Europe', ' Income '].mean()
print("the avrege income in Europe:",avg,"$ \n")


#26.what is the avrege income in pacific?
avg=data.loc[data['Region'] == 'Pacific', ' Income '].mean()
print("the avrege income in pacific:",avg,"$ \n")



#27.What is the most common Marital Status in dataset?
status=data['Marital Status'].mode().iloc[0]
print("most common marital status: ",status,"\n")


#28.what is the avrege income for single ?
avg_status=data[data["Marital Status"]=="single"] [" Income "].mean()
print("the avrege income for single:",avg_status,"$\n")



#28.what is the avrege income for married ?
avg_status=data[data["Marital Status"]=="married"] [" Income "].mean()

print("the avrege income for married:",avg_status,"$\n")




income_by_cars = data.groupby("Cars")[" Income "].mean()


plt.bar(income_by_cars.index, income_by_cars.values)
plt.xlabel("Number of Cars")
plt.ylabel("Mean Income")
plt.show()


sns.boxplot(data=data[" Income "])


plt.title('income')
plt.xlabel('income')
plt.ylabel('Value')


plt.show()
sns.boxplot(data=data["Age"])


plt.title('age')
plt.xlabel('age')
plt.ylabel('Value')


plt.show()


x = pd.pivot_table(index='Education', columns='Occupation', values='ID', aggfunc='count', data=data)


sns.heatmap(x, annot=True)
plt.show()

sns.countplot(x="Occupation", data=data)
plt.show()

corr = data.corr(numeric_only=True)

sns.heatmap(corr, annot=True)
plt.show()

plt.scatter(data["Age"], data[" Income "])
plt.xlabel("Age")
plt.ylabel("Income")
plt.show()


data.to_csv('pafter.csv', index=False)








from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def fix_Marital(a):
    if( a=="married" ):
        return "1"
    else:
        return "0"
data["Marital Status"]=data["Marital Status"].apply(fix_Marital).astype(int)

def fix_gender(a):
    if  a=="male" :
        return "1"
    else :
        return "0"
data['Gender'] = data['Gender'].apply(fix_gender).astype(int)


data=data[[" Income ","Age","Children","Cars","Purchased Bike","Marital Status","Gender"]]
bins = [0, 50000, 100000, np.inf]
labels = ["Low", "Medium", "High"]
data["Income_cat"] = pd.cut(data[" Income "], bins=bins, labels=labels)

# Drop the original income column and keep the binned income column
data = data.drop(columns=[" Income "])

# Convert the income category to numeric values
data["Income_cat"] = pd.factorize(data["Income_cat"])[0]

# Split the data into features (X) and target variable (y)
X = data.drop(columns=["Income_cat"])
y = data["Income_cat"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a machine learning model
logistic_regression_model = LogisticRegression(max_iter=1000)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_regression_model.fit(X_train_scaled, y_train)

# Use the trained model to predict on the test data
y_pred = logistic_regression_model.predict(X_test_scaled)

# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)