import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# reading the actual csv file
train = pd.read_csv('D:/Downloads/finance.export.order-item-transaction 2021-03-11.csv')

# printing the read file appropriately
print(train.head())

# making a heat map to see which columns possess null values and how many null values as well
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Null Columns')
plt.figure()

# we see that Comment and PaymentRefId are totally empty by looking at the heatmap

''' so from displaying our data and by looking at the heatmap, let's get rid of all the null columns 
    as well as the columns with no information relevant to our task '''

train.drop(['Comment', 'PaymentRefId', 'Transaction Date', 'Transaction Type', 'Fee Name', 'Transaction Number',
            'Details', 'Lazada SKU', 'Statement', 'Paid Status', 'Order Item No.', 'Order Item Status',
            'Shipping Provider', 'Shipping Speed', 'Shipment Type', 'Reference'], axis=1, inplace=True)

# printing the read file appropriately to see that neither the null columns exist anymore, nor the unneeded ones
print(train.head())

# making a heat map to now see that no columns have null values
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Null Columns (Removed)')
plt.figure()

''' There could now still be redundant data, to look into that let's have a look at the number of times
    the same value is repeated in such columns which appear to be redundant and we cannot go through them manually '''

sns.set_style('darkgrid')
sns.countplot(x='WHT Amount', data=train)
plt.title('Counts of values in WHT Amount')
plt.figure()

''' As suspected, WHT Amount ony has the values 0, therefore let's drop it, but before that let's also look at
    WHT included in Amount, which must be following the same trend since it is result of WHT Amount '''

sns.set_style('darkgrid')
sns.countplot(x='WHT included in Amount', data=train)
plt.title('Counts of data in WHT included in Amount')
# plt.show()
plt.figure()

# Yes, our intuition was correct and both follow and identical trend and are unneeded, so let's drop them too
train.drop(['WHT Amount', 'WHT included in Amount'], axis=1, inplace=True)

# printing the read file appropriately to see that neither the null columns exist anymore, nor the unneeded ones finally
print(train.head())

# creating a new dataframe to which I shall save my desired results
new_dataframe = pd.DataFrame(columns=['Seller SKU', 'Order Number', 'Profit'])

''' creating a sorted version my original dataframe so as to avoid any repeated order numbers that have different
    order numbers between them from being considered different order numbers as the given data is nonideal '''
train = train.sort_values(by=['Order No.'])

# creating a variable, and a temporary variable to be used in the for loop
profit = 0
order_temp = train.iloc[0]['Order No.']

# for loop traversing through the values of Order No. and comparing the initial one with the next
for i in range(train.shape[0]):
    # if initial order number matches the next one, save that to the profit variable
    if train.iloc[i]['Order No.'] == order_temp:
        profit = profit + train.iloc[i]['Amount'] - train.iloc[i]['VAT in Amount']
    # otherwise assign a seller sku of the previous order to a variable
    # create a new dataframe with the new seller sku, Order No., and profit
    # append the new dataframe created earlier with this new (updated) dataframe
    # assign a new Order No. to the temporary variable and reset the profit variable
    else:
        seller_SKU = train.iloc[i - 1]['Seller SKU']
        updated_dataframe = pd.DataFrame({'Seller SKU': [seller_SKU], 'Order Number': [order_temp], 'Profit': [profit]})
        new_dataframe = new_dataframe.append(updated_dataframe, ignore_index=True)
        order_temp = train.iloc[i]['Order No.']
        profit = 0

# printing the new dataframe
print(new_dataframe.head())
# create a dataframe file of the new dataframe
new_dataframe.to_csv('D:/Downloads/Profit_From_Each_Sold_Order.csv', index=False)

# counting the quantity of each sold seller SKU
seller_sku_count = new_dataframe['Seller SKU'].value_counts()
print(seller_sku_count)
# getting the list of seller SKUs from the series of seller_sku_count
x_axis = seller_sku_count.index.tolist()
# getting the list of quantity each seller SKU sold from the series of seller_sku_count
y_axis = seller_sku_count.values
# plotting a bargraph with a darkgrid that denotes the quantity of each sold seller SKU
sns.set_style('darkgrid')
sns.barplot(x_axis, y_axis)
plt.title('Quantity of Each Sold Seller SKU')
plt.figure()

# creating a new dataframe to with each Seller SKU next its profit
new_dataframe_2 = pd.DataFrame({'Seller SKU': x_axis, 'Profit': np.zeros(seller_sku_count.shape[0])})

# traversing through all elements of the old data frame
for i in range(new_dataframe.shape[0]):
    # traversing through all elements of the new data frame
    for j in range(new_dataframe_2.shape[0]):
        # if old data frame's seller SKU matches with new dataframe
        if new_dataframe.iloc[i]['Seller SKU'] == new_dataframe_2.iloc[j]['Seller SKU']:
            # put the profit of the old dataframe in a temporary variable
            profit_temp = new_dataframe.iloc[i]['Profit']
            # then update the profit of the new dataframe with its current profit + profit of older dataframe
            new_dataframe_2.at[j, 'Profit'] = new_dataframe_2.at[j, 'Profit'] + profit_temp

new_dataframe_2.to_csv('D:/Downloads/Profit_From_Each_Sold_SKU.csv', index=False)
x_axis = new_dataframe_2['Seller SKU'].tolist()
y_axis = new_dataframe_2['Profit'].tolist()
sns.set_style('darkgrid')
sns.barplot(x_axis, y_axis)
plt.title('Profit of Each Sold SKU')
plt.show()
plt.figure()


# profit = (train['Amount'] - train['VAT in Amount']).to_numpy()
# print(profit)
#
# product_name = train['Seller SKU'].to_numpy()
# print(product_name)
#
# order_number = train['Order No.'].to_numpy()
#
# individual_profit = list()
# individual_order_number = list()
# individual_product_name = list()
#
# individual_product_name.append(product_name[0])
# individual_order_number.append(order_number[0])
#
# individual_profit = list(set(order_number))
#
# print((individual_profit))
#
# for i in range(len(order_number)-1):
#     if order_number[i] == order_number[i+1]:
#         #a = profit[i] + profit[i+1]
#         #individual_profit[i] = a
#         None
#     else:
#         individual_order_number.append(order_number[i])
#         individual_product_name.append(product_name[i])
#         individual_profit.append(profit[i+1])
#
# print((individual_order_number))
# print(len(individual_product_name))
