# Q2: Generate association rules using FP-Growth algorithm on Online Retail dataset
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Load the Online Retail Excel file
df = pd.read_excel("Online Retail.xlsx")
df = df[df['InvoiceNo'].notnull() & df['Description'].notnull()]
df = df[df['Quantity'] > 0]

# Group products by transaction (InvoiceNo)
basket = df.groupby(['InvoiceNo'])['Description'].apply(list).reset_index(name='Items')

# One-hot encode the transaction data
te = TransactionEncoder()
te_ary = te.fit(basket['Items']).transform(basket['Items'])
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Apply FP-Growth algorithm with minimum support of 0.01
frequent_itemsets = fpgrowth(df_encoded, min_support=0.01, use_colnames=True)

# Generate association rules with minimum confidence of 0.5
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Display all rules with antecedents, consequents, support, lift, and confidence
for _, row in rules.iterrows():
    print(f"{set(row['antecedents'])} -> {set(row['consequents'])}, "
          f"Support={row['support']:.2f}, Lift={row['lift']:.2f}, Confidence={row['confidence']:.2f}")
