import pandas as pd
from ucimlrepo import fetch_ucirepo
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx
import matplotlib.pyplot as plt

# Step 1: Load Mushroom Dataset
mushroom = fetch_ucirepo(id=73)  # Mushroom dataset

# Step 2: Convert to DataFrame
df = mushroom.data.features
print("Sample data:\n", df.head())

# Step 3: One-hot encode categorical data
df_encoded = pd.get_dummies(df)

# Step 4: Apply Apriori algorithm
frequent_itemsets = apriori(df_encoded, min_support=0.3, use_colnames=True)

# Step 5: Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Step 6: Display Results
print("\n✅ Frequent Itemsets:")
print(frequent_itemsets.sort_values(by='support', ascending=False).head(10))

print("\n✅ Strong Association Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

# Step 7: Visualize Top 10 Association Rules as a Network Graph
top_rules = rules.sort_values(by="lift", ascending=False).head(10)

# Create graph
G = nx.DiGraph()

# Add nodes and edges
for _, row in top_rules.iterrows():
    for antecedent in row['antecedents']:
        for consequent in row['consequents']:
            G.add_edge(antecedent, consequent, weight=row['lift'], label=f"{row['confidence']:.2f}")

# Draw the graph
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=0.7)

nx.draw(G, pos, with_labels=True, node_color='turquoise', node_size=6000, font_size=8, font_weight='bold', arrows=True)
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

plt.title("Top 10 Association Rules (Confidence Shown on Edges)", fontsize=8)
plt.show()
