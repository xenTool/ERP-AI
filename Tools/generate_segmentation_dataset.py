import pandas as pd
import numpy as np

np.random.seed(42) # Für reproduzierbare Zufallswerte

n_customers = 50   # Es gibt 50 verschiedene Kunden (CUST_001 ... CUST_050)
n_orders = 500     # Insgesamt 500 Bestellungen

# Kunden-IDs wie "CUST_001", "CUST_002", ...
customer_ids = [f"CUST_{i:03d}" for i in range(1, n_customers + 1)]

# Jeder Bestellung wird zufällig ein Kunde zugeordnet
customers = np.random.choice(customer_ids, size=n_orders)

# Eindeutige Bestell-IDs wie "ORD_0001", "ORD_0002", ...
order_ids = [f"ORD_{i:04d}" for i in range(1, n_orders + 1)]

# Alle Bestellungen finden innerhalb eines Jahres statt (365 Tage
dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(
    np.random.randint(0, 365, n_orders), unit="D"
)
# Nettobeträge mit verschiedenen Kundensegmenten (z. B. Budget-, Normal-, Premium-Käufer)
segments = np.random.choice(["budget", "normal", "premium"], size=n_orders, p=[0.4, 0.4, 0.2])

# Preis (net_amount) abhängig vom Segment erzeugen
net_amount = np.where(
    segments == "budget", np.random.uniform(10, 50, n_orders),
    np.where(segments == "normal", np.random.uniform(50, 200, n_orders),
             np.random.uniform(200, 600, n_orders))
)

# Ca. 10 % der Bestellungen werden als Retoure markiert
returned = np.random.choice([0, 1], size=n_orders, p=[0.9, 0.1])

df = pd.DataFrame({
    "customer_id": customers,
    "order_id": order_ids,
    "order_date": dates,
    "net_amount": net_amount.round(2),  # auf 2 Dezimalstellen runden
    "returned": returned
})

# Datei speichern
df.to_csv("examples/example_customers.csv", index=False)

print(df.head()) # Ausgabe zur Kontrolle (erste 5 Zeilen)
