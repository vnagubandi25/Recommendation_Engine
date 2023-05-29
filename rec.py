
import pandas as pd
import numpy as np
from sklearn import model_selection, metrics, preprocessing
import torch
import torch.nn as nn
# import matplotlib.pyplot as plt 
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import Dataset


import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv("/data/venka/cse272/hw2/data/model_data.csv")
print(df.overall.unique())

class AutoDataset(Dataset):
    def __init__(self, users, items, ratings):
        self.users = users
        self.item = items
        self.ratings = ratings
    def __len__(self):
        return len(self.users)
    def __getitem__(self, item):
        users = self.users[item]
        item = self.item[item]
        ratings = self.ratings[item]
        return torch.tensor(users, dtype=torch.long).to(device), torch.tensor(item, dtype=torch.long).to(device), torch.tensor(ratings, dtype=torch.long).to(device)
        
    
class RecModel(nn.Module):
    def __init__(self, n_users, n_autos):
        super().__init__()
        self.user_embed = nn.Embedding(n_users, 32)
        self.auto_embed = nn.Embedding(n_autos, 32)
        self.out = nn.Linear(64, 1)

    def forward(self, users, autos):
        user_embeds = self.user_embed(users)
        auto_embeds = self.auto_embed(autos)
        output = torch.cat([user_embeds, auto_embeds], dim=1)
        output = self.out(output)
        return output
    

lbl_user = preprocessing.LabelEncoder()
lbl_movie = preprocessing.LabelEncoder()
df.reviewerID = lbl_user.fit_transform(df.reviewerID.values)
df.asin = lbl_movie.fit_transform(df.asin.values)

df_train, df_valid = model_selection.train_test_split(
    df, test_size=0.2, random_state=42
)

train_dataset = AutoDataset(
   df_train.reviewerID.values,
    df_train.asin.values,
    df_train.overall.values
)

valid_dataset = AutoDataset(
   df_valid.reviewerID.values,
    df_valid.asin.values,
    df_valid.overall.values
)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=2048,
                          shuffle=True,
                          ) 

validation_loader = DataLoader(dataset=valid_dataset,
                          batch_size=2048,
                          shuffle=True,
                          ) 

model = RecModel(
    n_users=len(lbl_user.classes_),
    n_autos=len(lbl_movie.classes_),
).to(device)


def train(model,train_loader):
    optimizer = torch.optim.Adam(model.parameters())  
    sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)

    loss_func = nn.MSELoss()

    epochs = 10
    total_loss = 0
    step_cnt = 0
    all_losses_list = [] 

    model.train() 
    for epoch in range(epochs):
        for batch  in train_loader:
            user_ids, item_ids, ratings = batch
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            ratings = ratings.view(-1,1).to(torch.float32).to(device)
            output = model(user_ids,
                        item_ids
                        ) 
            rating = ratings
            loss = loss_func(output,ratings)
            total_loss = total_loss + loss.sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def evaluate(model, data_loader,floor_or_ceil):
    model.eval()
    true_labels = []
    predictions = []
    with torch.no_grad():
        for batch in data_loader:
            user_ids, item_ids, ratings = batch
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            ratings = ratings.view(-1,1).to(torch.float32).to(device)
            output = model(user_ids,item_ids) 
            true_labels.extend(ratings.cpu().numpy().flatten())
            if floor_or_ceil == 0:
                predictions.extend(torch.floor(output).cpu().numpy().flatten())
            else:
                predictions.extend(torch.ceil(output).cpu().numpy().flatten())
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    fmeasure = f1_score(true_labels, predictions, average='weighted')
    rmse = mean_squared_error(true_labels, predictions, squared=False)
    mae =  mean_absolute_error(true_labels, predictions,)
    return precision, recall, fmeasure, rmse, mae

def recommend_items(model, n_items=10, n_users=10):

    user_ids = df_valid['reviewerID'].unique()[0:n_users]
    recommendations = {}
    user_item_pairs = []
    item_indices = []
    
    for user_id in user_ids:
        purchased_items = df_train.loc[df_train['reviewerID'] == user_id, 'asin'].unique()
        all_items = df['asin'].unique()
        not_purchased_items = np.setdiff1d(all_items, purchased_items)
        user_item_pairs.extend(list(zip([user_id]*len(not_purchased_items), not_purchased_items)))
        item_indices.append((len(item_indices[-1]) if len(item_indices) > 0 else 0) + np.arange(len(not_purchased_items)))

    user_item_pairs_tensor = torch.tensor(user_item_pairs, dtype=torch.long)
    user_ids_tensor = user_item_pairs_tensor[:, 0]
    item_ids_tensor = user_item_pairs_tensor[:, 1]
    predicted_ratings = model(user_ids_tensor.to(device), item_ids_tensor.to(device)).detach().cpu().numpy().flatten()
    
    for i, user_id in enumerate(user_ids):
        predicted_ratings_user = predicted_ratings[item_indices[i]]
        recommended_item_ids = not_purchased_items[np.argsort(-predicted_ratings_user)][:n_items]
        recommended_item_labels = lbl_movie.inverse_transform(recommended_item_ids)
        original_user_id = lbl_user.inverse_transform([user_id])[0]
        recommendations[original_user_id] = recommended_item_labels
    return recommendations


train(model=model,train_loader=train_loader)

train_precision, train_recall, train_fmeasure, train_rmse, train_mae = evaluate(model, train_loader,1)
print("For ceiling on Train")
print(f'Precision: {train_precision:.4f}')
print(f'Recall: {train_recall:.4f}')
print(f'F-measure: {train_fmeasure:.4f}')
print(f'rmse: {train_rmse:.4f}')
print(f'mae: {train_mae:.4f}')
print()
print("For ceiling on validation")
precision, recall, fmeasure,rmse, mae = evaluate(model, validation_loader,1)
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F-measure: {fmeasure:.4f}')
print(f'rmse: {rmse:.4f}')
print(f'mae: {mae:.4f}')

print()
print()
print()

print("For floor on Train")
train_precision, train_recall, train_fmeasure, train_rmse, train_mae = evaluate(model, train_loader,0)
print(f'Precision: {train_precision:.4f}')
print(f'Recall: {train_recall:.4f}')
print(f'F-measure: {train_fmeasure:.4f}')
print(f'rmse: {train_rmse:.4f}')
print(f'mae: {train_mae:.4f}')
print()
print("For floor on Validation")

precision, recall, fmeasure,rmse, mae = evaluate(model, validation_loader,0)
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F-measure: {fmeasure:.4f}')
print(f'rmse: {rmse:.4f}')
print(f'mae: {mae:.4f}')

recommendations_fast = recommend_items(model,10,10)


