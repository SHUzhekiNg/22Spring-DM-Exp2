from myfunc_cop import *

num_classes = 17
embedding_size = 20

dtype = torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data, vocab_size = data_loader(is_train=False)

tcnn = TextCNN(vocab_size=vocab_size,
               embedding_size=embedding_size,
               num_classes=num_classes)
model = tcnn.to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Training
for epoch in tqdm(range(80)):
    if (epoch + 1) % 2 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
    for batch_x, batch_y in train_data:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Test
test_data, vocab_size = data_loader(is_train=False)
model = model.eval()
correct = 0
total = 0
for batch_x, batch_y in test_data:
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    predict = model(batch_x).data.max(1, keepdim=True)[1]
    # print(predict.squeeze(), batch_y)
    # print(predict.squeeze() == batch_y)
    total += batch_y.size(0)
    correct += (predict.squeeze() == batch_y).sum().item()

# print(total)
print('Accuracy of the model on the test data: %d %%' % (100 * correct / total))

# test_text = 'i hate me'
# tests = [[tcnn.word2idx[n] for n in test_text.split()]]
# test_batch = torch.LongTensor(tests).to(device)
# Predict

# predict = model(batch_x).data.max(1, keepdim=True)[1]
# if predict[0][0] == 0:
#     print(test_text,"is Bad Mean...")
# else:
#     print(test_text,"is Good Mean!!")