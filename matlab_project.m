
% veriyi çekme
data = readtable('audi.csv');

% dataframe içindeki transmission sütununu kategorik olarak işaretleme
data.transmission = categorical(data.transmission);

% Kategorik değerleri numeric değerlere dönüştürme
data.numeric_transmission = grp2idx(data.transmission);

% dataframe içindeki fuelType sütununu kategorik olarak işaretleme
data.fuelType = categorical(data.fuelType);

% Kategorik değerleri numeric değerlere dönüştürme
data.numeric_fuelType = grp2idx(data.fuelType);

% Sonuçları gösterme
disp('Orijinal Veri:');
disp(data);

% gereksiz sutünu dataframe den silme
data = removevars(data, 'model');
data = removevars(data, 'transmission');
data = removevars(data, 'fuelType');

disp(data);

% Y değişkenini seç
y = data.price;

% Diğer bütün sütunları X olarak al
X = data(:, ~strcmp(data.Properties.VariableNames, 'price'));

% Rastgele veri bölme için indeksleri oluştur
cv = cvpartition(size(data, 1), 'HoldOut', 0.2); % 80% eğitim, 20% test

% Eğitim ve test kümelerini oluştur
trainIdx = training(cv);
testIdx = test(cv);

% Eğitim ve test verilerini ayır
X_train = X(trainIdx,:);
X_test = X(testIdx,:);
y_train = y(trainIdx);
y_test = y(testIdx);

% Lineer regresyon modelini oluştur
lm = fitlm(X_train, y_train);

% Model özetini görüntüle
disp(lm);

% Modeli test et
y_predicted = predict(lm, X_test);

% Hata metriklerini hesapla
mse = mean((y_test - y_predicted).^2);
mae = mean(abs(y_test - y_predicted));
r_squared = 1 - sum((y_test - y_predicted).^2) / sum((y_test - mean(y_test)).^2);

% Sonuçları görüntüle
fprintf('Mean Squared Error (MSE): %.4f\n', mse);
fprintf('Mean Absolute Error (MAE): %.4f\n', mae);
fprintf('R-squared: %.4f\n', r_squared);
