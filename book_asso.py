# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 13:49:36 2020

@author: Harish
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
import matplotlib.pyplot as plt
book=pd.read_csv("book.csv")
frequent_itemsets = apriori(book, min_support=0.005, max_len=3,use_colnames = True)
frequent_itemsets.sort_values('support',ascending = False,inplace=True)
plt.bar(x = list(range(1,5)),height = frequent_itemsets.support[1:5],color='rgmyk');plt.xticks(list(range(1,5)),frequent_itemsets.itemsets[1:5])
plt.xlabel('item-sets');plt.ylabel('support')
rules1 = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules1.shape
#1054 rules

rules1.sort_values('lift',ascending = False,inplace=True)
#to eliminate redundancy
def to_list(i):
    return (sorted(list(i)))


ma_X = rules1.antecedents.apply(to_list)+rules1.consequents.apply(to_list)


ma_X = ma_X.apply(sorted)

rules1_sets = list(ma_X)

unique_rules1_sets = [list(m) for m in set(tuple(i) for i in rules1_sets)]
index_rules1 = []
for i in unique_rules1_sets:
    index_rules1.append(rules1_sets.index(i))
    
rules1_no_redudancy  = rules1.iloc[index_rules1,:]
rules1_no_redudancy.shape
#212 Rules
rules1_no_redudancy.sort_values('lift',ascending=False).head(10)
#top 10 Rules


##################Changine Support and min length############

frequent_itemsets2 = apriori(book, min_support=0.01, max_len=4,use_colnames = True)
frequent_itemsets2.sort_values('support',ascending = False,inplace=True)
plt.bar(x = list(range(1,5)),height = frequent_itemsets2.support[1:5],color='rgmyk');plt.xticks(list(range(1,5)),frequent_itemsets2.itemsets[1:5])
plt.xlabel('item-sets');plt.ylabel('support')
rules2 = association_rules(frequent_itemsets2, metric="lift", min_threshold=1)
rules2.shape
#3762 rules


#eliminate redundancy

def to_list(i):
    return (sorted(list(i)))


ma_X = rules2.antecedents.apply(to_list)+rules2.consequents.apply(to_list)


ma_X = ma_X.apply(sorted)

rules2_sets = list(ma_X)

unique_rules2_sets = [list(m) for m in set(tuple(i) for i in rules2_sets)]
index_rules2 = []
for i in unique_rules2_sets:
    index_rules2.append(rules2_sets.index(i))
    
rules2_no_redudancy  = rules2.iloc[index_rules2,:]
rules2_no_redudancy.shape
#396 rules
rules2_no_redudancy.sort_values('lift',ascending=False).head(10)
