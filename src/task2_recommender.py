import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

transactions = [
    ["pasta", "olive oil", "canned tomatoes", "garlic", "onion"],
    ["pasta", "olive oil", "canned tomatoes", "parmesan"],
    ["pasta", "olive oil", "garlic", "parmesan"],
    ["pasta", "canned tomatoes", "garlic", "onion"],
    ["pasta", "olive oil", "canned tomatoes"],
    ["bread", "butter", "jam"],
    ["bread", "butter", "eggs"],
    ["bread", "jam", "peanut butter"],
    ["milk", "cereal", "banana"],
    ["milk", "coffee", "sugar"],
    ["coffee", "milk", "sugar"],
    ["tea", "biscuits"],
    ["tea", "milk", "biscuits"],
    ["apple", "banana", "yogurt"],
    ["banana", "yogurt", "granola"],
    ["rice", "chickpeas", "canned tomatoes", "onion"],
    ["rice", "beans", "canned tomatoes"],
    ["olive oil", "garlic", "onion", "canned tomatoes"],
    ["pasta", "olive oil", "canned tomatoes", "basil"],
    ["pasta", "olive oil", "canned tomatoes", "garlic", "basil"],
]

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
basket_df = pd.DataFrame(te_array, columns=te.columns_)

frequent_itemsets = apriori(basket_df, min_support=0.15, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

print("Frequent Itemsets:\n")
print(frequent_itemsets)

print("\nAssociation Rules:\n")
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]])

input_basket = {"pasta", "olive oil"}
recommendation_scores = {}

for _, row in rules.iterrows():
    antecedents = set(row["antecedents"])
    consequents = set(row["consequents"])
    confidence = row["confidence"]

    if antecedents.issubset(input_basket):
        for item in consequents:
            if item not in input_basket:
                if item not in recommendation_scores:
                    recommendation_scores[item] = confidence
                else:
                    recommendation_scores[item] = max(recommendation_scores[item], confidence)

sorted_recommendations = sorted(
    recommendation_scores.items(),
    key=lambda x: x[1],
    reverse=True
)

print("\nInput basket:", input_basket)
print("Recommended extra items:")
for item, score in sorted_recommendations:
    print(f"{item} (confidence={score:.2f})")