from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    features = [
        'GDP_Growth', 'Inflation', 'Debt_to_GDP', 'Exchange_Rate',
        'Foreign_Reserves', 'Current_Account_Balance',
        'Debt_Service_Exports', 'Political_Stability', 'Interest_Rate'
    ]
    X = df[features]
    y = df['Default']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, scaler
