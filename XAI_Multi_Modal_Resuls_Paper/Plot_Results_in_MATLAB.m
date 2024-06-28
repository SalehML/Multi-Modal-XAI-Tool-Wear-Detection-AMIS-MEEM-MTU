clear
clc
close all

mat_full = readtable('Matrix_For_Test_data.xlsx');

mat_full = mat_full{2:3, 2:3};

acc_full = (sum(diag(mat_full)) / 28) * 100;

mat_CNN_I = readtable('Matrix_For_Test_data_CNN_I.xlsx');

mat_CNN_I = mat_CNN_I{2:3, 2:3};

acc_CNN_I = (sum(diag(mat_CNN_I)) / 28) * 100;

mat_CNN_II = readtable('Matrix_For_Test_data_CNN_II.xlsx');

mat_CNN_II = mat_CNN_II{2:3, 2:3};

acc_CNN_II = (sum(diag(mat_CNN_II)) / 28) * 100;

mat_RNN = readtable('Matrix_For_Test_data_RNN.xlsx');

mat_RNN = mat_RNN{2:3, 2:3};

acc_RNN = (sum(diag(mat_RNN)) / 28) * 100;

figure(1)
imagesc(mat_full);
colorbar;
title(['Accuracy: ', num2str(acc_full), '%']);
xlabel('Predicted Label', 'FontSize', 16);
ylabel('True Label', 'FontSize', 16);
textStrings = num2str(mat_full(:),'%d');
textStrings = strtrim(cellstr(textStrings));
[x, y] = meshgrid(1:size(mat_full,1), 1:size(mat_full,2));
hStrings = text(x(:), y(:), textStrings(:), 'HorizontalAlignment', 'center');
xticks(1:2);
xticklabels({'Unworn', 'Worn'});
yticks(1:2);
yticklabels({'Unworn', 'Worn'});
set(hStrings, 'FontSize', 20);

figure(2)

imagesc(mat_CNN_I);
colorbar;
title(['Accuracy: ', num2str(acc_CNN_I), '%']);
xlabel('Predicted Label', 'FontSize', 16);
ylabel('True Label', 'FontSize', 16);
textStrings = num2str(mat_CNN_I(:),'%d');
textStrings = strtrim(cellstr(textStrings));
[x, y] = meshgrid(1:size(mat_CNN_I,1), 1:size(mat_CNN_I,2));
hStrings = text(x(:), y(:), textStrings(:), 'HorizontalAlignment', 'center');
xticks(1:2);
xticklabels({'Unworn', 'Worn'});
yticks(1:2);
yticklabels({'Unworn', 'Worn'});
set(hStrings, 'FontSize', 20);


figure(3)

imagesc(mat_CNN_II);
colorbar;
title(['Accuracy: ', num2str(acc_CNN_II), '%']);
xlabel('Predicted Label', 'FontSize', 16);
ylabel('True Label', 'FontSize', 16);
textStrings = num2str(mat_CNN_II(:),'%d');
textStrings = strtrim(cellstr(textStrings));
[x, y] = meshgrid(1:size(mat_CNN_II,1), 1:size(mat_CNN_II,2));
hStrings = text(x(:), y(:), textStrings(:), 'HorizontalAlignment', 'center');
xticks(1:2);
xticklabels({'Unworn', 'Worn'});
yticks(1:2);
yticklabels({'Unworn', 'Worn'});
set(hStrings, 'FontSize', 20);

figure(4)

imagesc(mat_RNN);
colorbar;
title(['Accuracy: ', num2str(acc_RNN), '%']);
xlabel('Predicted Label', 'FontSize', 16);
ylabel('True Label', 'FontSize', 16);
textStrings = num2str(mat_RNN(:),'%d');
textStrings = strtrim(cellstr(textStrings));
[x, y] = meshgrid(1:size(mat_RNN,1), 1:size(mat_RNN,2));
hStrings = text(x(:), y(:), textStrings(:), 'HorizontalAlignment', 'center');
xticks(1:2);
xticklabels({'Unworn', 'Worn'});
yticks(1:2);
yticklabels({'Unworn', 'Worn'});
set(hStrings, 'FontSize', 20);


line = [0 1; 0 1];

True_Performance_1 = [0 0; 0 1];

True_Performance_2 = [0 1; 1 1];
ROC_Full = readtable('ROF_full_model.xlsx');

ROC_Full = ROC_Full{2:3, 2:4};

ROC_CNN_I = readtable('ROF_CNN_I_model.xlsx');

ROC_CNN_I = ROC_CNN_I{2:3, 2:4};

ROC_CNN_II = readtable('ROF_CNN_II_model.xlsx');

ROC_CNN_II = ROC_CNN_II{2:3, 2:4};

ROC_RNN = readtable('ROF_RNN_model.xlsx');

ROC_RNN = ROC_RNN{2:3, 2:4};

figure(5)
plot(ROC_Full(2,:), ROC_Full(1,:), 'r', LineWidth = 2.5)
hold on 
plot(line(1,:), line(2,:), 'k--', LineWidth = 2.5)
hold on 
plot(True_Performance_1(1,:), True_Performance_1(2,:), 'b--', LineWidth = 2.5)
hold on 
plot(True_Performance_2(1,:), True_Performance_2(2,:), 'b--', LineWidth = 2.5)
legend('Trained Model Performance', 'Untrained Model Performance', 'Ideal Model Performance', 'FontSize', 14)
xlabel('FPR', 'FontSize', 14)
ylabel('TPR', 'FontSize', 14)

figure(6)
plot(ROC_CNN_I(2,:), ROC_CNN_I(1,:), 'r', LineWidth = 2.5)
hold on 
plot(line(1,:), line(2,:), 'k--', LineWidth = 2.5)
hold on 
plot(True_Performance_1(1,:), True_Performance_1(2,:), 'b--', LineWidth = 2.5)
hold on 
plot(True_Performance_2(1,:), True_Performance_2(2,:), 'b--', LineWidth = 2.5)
legend('Trained Model Performance', 'Untrained Model Performance', 'Ideal Model Performance', 'FontSize', 14)
xlabel('FPR', 'FontSize', 14)
ylabel('TPR', 'FontSize', 14)

figure(7)
plot(ROC_CNN_II(2,:), ROC_CNN_II(1,:), 'r', LineWidth = 2.5)
hold on 
plot(line(1,:), line(2,:), 'k--', LineWidth = 2.5)
hold on 
plot(True_Performance_1(1,:), True_Performance_1(2,:), 'b--', LineWidth = 2.5)
hold on 
plot(True_Performance_2(1,:), True_Performance_2(2,:), 'b--', LineWidth = 2.5)
legend('Trained Model Performance', 'Untrained Model Performance', 'Ideal Model Performance', 'FontSize', 14)
xlabel('FPR', 'FontSize', 14)
ylabel('TPR', 'FontSize', 14)

figure(8)
plot(ROC_RNN(2,:), ROC_RNN(1,:), 'r', LineWidth = 2.5)
hold on 
plot(line(1,:), line(2,:), 'k--', LineWidth = 2.5)
hold on 
plot(True_Performance_1(1,:), True_Performance_1(2,:), 'b--', LineWidth = 2.5)
hold on 
plot(True_Performance_2(1,:), True_Performance_2(2,:), 'b--', LineWidth = 2.5)
legend('Trained Model Performance', 'Untrained Model Performance', 'Ideal Model Performance', 'FontSize', 14)
xlabel('FPR', 'FontSize', 14)
ylabel('TPR', 'FontSize', 14)

% epochs = linspace(1,10,10);
% 
% accuracy = readtable('Accuracy_Info_7.xlsx');
% 
% accuracy = accuracy{2:6, 2};
% 
% accuracy_val = readtable('Val_accuracy_Info_7.xlsx');
% 
% accuracy_val = accuracy_val{2:6, 2};
% 
% loss = readtable('Loss_Info_7.xlsx');
% 
% loss = loss{2:6, 2};
% 
% loss_val = readtable('Val_loss_Info_7.xlsx');
% 
% loss_val = loss_val{2:6, 2};
% 
% figure(9)
% plot(epochs, accuracy, 'r', LineWidth=2)
% hold on 
% plot(epochs, accuracy_val, 'b', LineWidth=2)
% xlabel('Training Iterations')
% ylabel('Network Accuracy Evolution During the Training')
% legend('Accuracy on Training Dataset', 'Accuracy on Test Dataset')
% 
% figure(10)
% plot(epochs, loss, 'r', LineWidth=2)
% hold on 
% plot(epochs, loss_val, 'b', LineWidth=2)
% xlabel('Training Iterations')
% ylabel('Loss Function Evolution During the Training')
% legend('Loss Function on Training Dataset', 'Loss Function on Test Dataset')

shap_latent_case_I = readtable('SHAP_Method_II_VAL\shap_latent_real_pd.xlsx');

shap_latent_case_I = shap_latent_case_I{2:4, 2:end};

latent_nums = [1 5 9];

for i = 1:3

    CNN_I_latent = abs(shap_latent_case_I(i, 1:15)) ;

    CNN_II_latent = abs(shap_latent_case_I(i, 16:30));

    RNN_latent = abs(shap_latent_case_I(i, 31:45));

    CNN_I_latent_mean = mean(CNN_I_latent);

    CNN_II_latent_mean = mean(CNN_II_latent);

    RNN_latent_mean = mean(RNN_latent);

    latents = [CNN_I_latent_mean, CNN_II_latent_mean, RNN_latent_mean];

    figure(i + 10)
    bar(latent_nums, latents)
    xlim([-2 13])
    xticks([1 5 9])
    xticklabels({'CNN I Latent Shap','CNN II Latent Shap','RNN Latent Shap'})
    xlabel('Latent Space Neural Representation of Submodels')
    ylabel('Mean Shap Value of each SubModel')

end

shap_latent_case_II = readtable('SHAP_Method_II_TEST\shap_latent_real_val_pd.xlsx');

shap_latent_case_II = shap_latent_case_II{2:end, 2:end};

latent_nums = [1 5 9];

for i = 1:7

    CNN_I_latent = abs(shap_latent_case_II(i, 1:15)) ;

    CNN_II_latent = abs(shap_latent_case_II(i, 16:30));

    RNN_latent = abs(shap_latent_case_II(i, 31:45));

    CNN_I_latent_mean = mean(CNN_I_latent);

    CNN_II_latent_mean = mean(CNN_II_latent);

    RNN_latent_mean = mean(RNN_latent);

    latents = [CNN_I_latent_mean, CNN_II_latent_mean, RNN_latent_mean];

    figure(i + 14)
    bar(latent_nums, latents)
    xlim([-2 13])
    xticks([1 5 9])
    xticklabels({'CNN I Latent Shap','CNN II Latent Shap','RNN Latent Shap'})
    xlabel('Latent Space Neural Representation of Submodels')
    ylabel('Mean Shap Value of each SubModel')
    
end